import numpy as np

import torch
import torch.nn.functional as F
import datetime
import time
from utils import *
from defense import *

from tqdm.auto import tqdm

from iba_helpers import *
import pandas as pd

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import csv_record

from termcolor import colored

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return output



def exponential_decay(init_val, decay_rate, t):
    return init_val*(1.0 - decay_rate)**t

def get_results_filename(poison_type, attack_method, model_replacement, project_frequency, defense_method, norm_bound, prox_attack, fixed_pool=False, model_arch="vgg9"):
    filename = "{}_{}_{}".format(poison_type, model_arch, attack_method)
    if fixed_pool:
        filename += "_fixed_pool" 
    
    if model_replacement:
        filename += "_with_replacement"
    else:
        filename += "_without_replacement"
    
    if attack_method == "pgd":
        filename += "_1_{}".format(project_frequency)
    
    if prox_attack:
        filename += "_prox_attack"

    if defense_method in ("norm-clipping", "norm-clipping-adaptive", "weak-dp"):
        filename += "_{}_m_{}".format(defense_method, norm_bound)
    elif defense_method in ("krum", "multi-krum", "rfa"):
        filename += "_{}".format(defense_method)
               
    filename += "_acc_results.csv"

    return filename


# ================================================
# =====PARAMETERS HELPER FUNCTIONS===============
# ================================================

def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="bad"):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    if mode == "bad":
        #pdb.set_trace()
        logger.info("===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "normal":
        logger.info("===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "avg":
        logger.info("===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))

    return norm_diff

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def _apply_l_inf_projection(model, model_original, pgd_eps):
    """Apply L-infinity projection for PGD attack.
    
    Args:
        model: Current model
        model_original: Original model parameters
        pgd_eps: Epsilon for PGD constraint
    """
    w = list(model.parameters())
    eta = 0.001  # adversarial learning rate
    
    for i in range(len(w)):
        w[i].data = w[i].data - eta * w[i].grad.data
        
        # Projection step
        m1 = torch.lt(torch.sub(w[i], model_original[i]), -pgd_eps)
        m2 = torch.gt(torch.sub(w[i], model_original[i]), pgd_eps)
        w1 = (model_original[i] - pgd_eps) * m1
        w2 = (model_original[i] + pgd_eps) * m2
        w3 = w[i] * (~(m1 + m2))
        wf = w1 + w2 + w3
        w[i].data = wf.data

def _apply_l2_projection(model, model_original, adv_optimizer, batch_idx, total_batches, 
                        project_frequency, pgd_eps):
    """Apply L2 projection for PGD attack.
    
    Args:
        model: Current model
        model_original: Original model parameters
        adv_optimizer: Adversarial optimizer
        batch_idx: Current batch index
        total_batches: Total number of batches
        project_frequency: Frequency of projection
        pgd_eps: Epsilon for PGD constraint
    """
    adv_optimizer.step()
    w = list(model.parameters())
    w_vec = parameters_to_vector(w)
    model_original_vec = parameters_to_vector(model_original)
    
    # Project on last iteration or at specified frequency
    if ((batch_idx % project_frequency == 0 or batch_idx == total_batches - 1) and 
        torch.norm(w_vec - model_original_vec) > pgd_eps):
        # Project back into norm ball
        w_proj_vec = (pgd_eps * (w_vec - model_original_vec) / 
                     torch.norm(w_vec - model_original_vec) + model_original_vec)
        # Plug w_proj back into model
        vector_to_parameters(w_proj_vec, w)

def fed_nova_aggregator(model, net_list, list_ni, device, list_ai):
    # https://github.com/Xtra-Computing/NIID-Bench/blob/692569f790af0f5908dba23a15d4d80ce7e7aec4/experiments.py#L375
    total_n = sum(list_ni)
    print(f"Aggregating models with FedNova with aggregation weight {list_ai}")
    print(f"List of sample size {list_ni}")
    
    weight_accumulator = {}
    for name, params in model.state_dict().items():
        weight_accumulator[name] = torch.zeros(params.size()).to(device)
    
    for net_index, net in enumerate(net_list):
        for name, data in net.state_dict().items():
            # weight_accumulator[name].add_( ( (data - model.state_dict()[name]) / list_ai[net_index] ) * (list_ni[net_index] / total_n))
            weight_accumulator[name].add_( torch.true_divide( (model.state_dict()[name] - data), list_ai[net_index] ) * list_ni[net_index] / total_n)
    
    coeff = 0.0
    for i in range(len(list_ai)):
        coeff += list_ai[i] * list_ni[i] / total_n
        # print(f"ai: {list_ai[i]} ni: {list_ni[i]} coeff: {coeff}")
    print(f"coeff: {coeff}")
    
    for name, params in model.state_dict().items():
        update_per_layer = coeff * weight_accumulator[name]
        # params.add_(coeff * update_per_layer)
        if params.type() != update_per_layer.type():
            params.sub_(update_per_layer.to(torch.int64))
        else:
            params.sub_(update_per_layer)
    return model

def fed_avg_aggregator(init_model, net_list, net_freq, device, model="lenet"):
    # import IPython
    # IPython.embed()
    
    weight_accumulator = {}
    
    for name, params in init_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params).float()
    
    for i in range(0, len(net_list)):
        diff = dict()
        for name, data in net_list[i].state_dict().items():
            # diff[name] = (data - model_server_before_aggregate.state_dict()[name]).cpu().detach().numpy()
            diff[name] = (data - init_model.state_dict()[name])
            try:
                weight_accumulator[name].add_(net_freq[i]  *  diff[name])
                # weight_accumulator[name].add_(0.1  *  diff[name])
                
            except Exception as e:
                print(e)
                import IPython
                IPython.embed()
                exit(0)
        # print(f"diff: {diff}")
    for idl, (name, data) in enumerate(init_model.state_dict().items()):
        update_per_layer = weight_accumulator[name] #  * self.conf["lambda"]
        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))            
        else:
            data.add_(update_per_layer)
            
    return init_model

def estimate_wg(model, device, train_loader, optimizer, epoch, log_interval, criterion):
    logger.info("Prox-attack: Estimating wg_hat")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# ================================================
# =====LOCAL TRAINING FUNCTIONS==================
# ================================================

def train_iba_local(model, atkmodel, tgtmodel, optimizer, atkmodel_optimizer, train_loader, criterion, 
                   atk_eps=0.001, attack_alpha=0.5, attack_portion=1.0, clip_image=None, 
                   target_transform=None, atk_train_epoch=1, atkmodel_train=False, device=None, 
                   pgd_attack=False, batch_idx=1, project_frequency=1, pgd_eps=None, 
                   model_original=None, adv_optimizer=None, proj="l_2", mask_grad_list=[], 
                   aggregator="fedavg", wg_hat=None, mu=0.1, local_e=0):
    """
    Train IBA (Invisible Backdoor Attack) model locally.
    
    Args:
        model: Main model to train
        atkmodel: Attack model for generating perturbations
        tgtmodel: Target model for attack generation
        optimizer: Optimizer for main model
        atkmodel_optimizer: Optimizer for attack model
        train_loader: Data loader for training
        criterion: Loss function
        atk_eps: Attack epsilon for perturbation magnitude
        attack_alpha: Weight for clean vs poisoned loss
        attack_portion: Portion of batch to attack
        clip_image: Function to clip images to valid range
        target_transform: Function to transform targets for attack
        atk_train_epoch: Number of epochs to train attack model
        atkmodel_train: Whether to train attack model
        device: Device to run on
        pgd_attack: Whether to use PGD attack
        batch_idx: Current batch index
        project_frequency: Frequency of PGD projection
        pgd_eps: Epsilon for PGD constraint
        model_original: Original model for PGD projection
        adv_optimizer: Adversarial optimizer for PGD
        proj: Projection type ("l_2" or "l_inf")
        mask_grad_list: List of gradient masks
        aggregator: Aggregation method ("fedavg", "fedprox", etc.)
        wg_hat: Global model for FedProx
        mu: FedProx regularization weight
        local_e: Local epoch number
    
    Returns:
        dict: Training metrics including loss and accuracies
    """
    tgtmodel.eval()
    wg_clone = copy.deepcopy(model)
    
    # Initialize metrics
    metrics = {
        'correct_clean': 0,
        'correct_poison': 0,
        'clean_size': 0,
        'poison_size': 0,
        'loss_list': []
    }
    
    if not atkmodel_train:
        _train_main_model(model, atkmodel, tgtmodel, optimizer, train_loader, criterion,
                         atk_eps, attack_alpha, attack_portion, clip_image, target_transform,
                         device, pgd_attack, batch_idx, project_frequency, pgd_eps,
                         model_original, adv_optimizer, proj, mask_grad_list,
                         aggregator, wg_hat, mu, local_e, metrics)
    else:
        _train_attack_model(model, atkmodel, atkmodel_optimizer, train_loader, criterion,
                           atk_eps, attack_portion, clip_image, target_transform,
                           device, metrics)
    
    # Calculate final metrics
    clean_acc = _calculate_accuracy(metrics['correct_clean'], metrics['clean_size'])
    poison_acc = _calculate_accuracy(metrics['correct_poison'], metrics['poison_size'])
    training_avg_loss = sum(metrics['loss_list']) / len(metrics['loss_list']) if metrics['loss_list'] else 0.0
    
    # Log results
    _log_training_results(training_avg_loss, clean_acc, poison_acc, atkmodel_train)
    
    del wg_clone
    return {
        'loss': training_avg_loss,
        'clean_acc': clean_acc,
        'poison_acc': poison_acc
    }


def _train_main_model(model, atkmodel, tgtmodel, optimizer, train_loader, criterion,
                     atk_eps, attack_alpha, attack_portion, clip_image, target_transform,
                     device, pgd_attack, batch_idx, project_frequency, pgd_eps,
                     model_original, adv_optimizer, proj, mask_grad_list,
                     aggregator, wg_hat, mu, local_e, metrics):
    """Train the main model with optional attack training."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(train_loader):
        bs = len(batch)
        data, targets = batch
        
        # Prepare data
        clean_images, clean_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
        poison_images, poison_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
        metrics['clean_size'] += len(clean_images)
        
        # Zero gradients
        _zero_gradients(optimizer, adv_optimizer, pgd_attack)
        
        # Forward pass and clean loss
        output = model(clean_images)
        loss_clean = loss_fn(output, clean_targets)
        
        # Add FedProx regularization if needed
        if aggregator == "fedprox":
            loss_clean = _add_fedprox_regularization(loss_clean, model, wg_hat, mu)
        
        if attack_alpha == 1.0:
            # Clean training only
            _perform_clean_training(model, optimizer, loss_clean, mask_grad_list, 
                                  pgd_attack, proj, model_original, adv_optimizer,
                                  batch_idx, len(train_loader), project_frequency, pgd_eps)
            
            # Update metrics
            pred = output.data.max(1)[1]
            metrics['loss_list'].append(loss_clean.item())
            metrics['correct_clean'] += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()
        else:
            # Attack training
            _perform_attack_training(model, atkmodel, tgtmodel, optimizer, adv_optimizer,
                                   clean_images, clean_targets, poison_images, poison_targets,
                                   output, loss_clean, attack_alpha, attack_portion, atk_eps,
                                   clip_image, target_transform, pgd_attack, proj, model_original,
                                   batch_idx, len(train_loader), project_frequency, pgd_eps,
                                   mask_grad_list, bs, device, metrics)


def _train_attack_model(model, atkmodel, atkmodel_optimizer, train_loader, criterion,
                       atk_eps, attack_portion, clip_image, target_transform, device, metrics):
    """Train the attack model."""
    model.eval()
    atkmodel.train()
    loss_fn = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        bs = data.size(0)
        data, target = data.to(device), target.to(device)
        metrics['poison_size'] += len(data)
        
        # Generate attack data
        noise = atkmodel(data) * atk_eps
        atkdata = clip_image(data + noise)
        atktarget = target_transform(target)
        
        if attack_portion < 1.0:
            atkdata = atkdata[:int(attack_portion * bs)]
            atktarget = atktarget[:int(attack_portion * bs)]
        
        # Forward pass and loss
        atkoutput = model(atkdata)
        loss_p = loss_fn(atkoutput, atktarget)
        
        # Backward pass
        atkmodel_optimizer.zero_grad()
        loss_p.backward()
        atkmodel_optimizer.step()
        
        # Update metrics
        pred = atkoutput.data.max(1)[1]
        metrics['correct_poison'] += pred.eq(atktarget.data.view_as(pred)).cpu().sum().item()
        metrics['loss_list'].append(loss_p.item())


def _zero_gradients(optimizer, adv_optimizer, pgd_attack):
    """Zero gradients for optimizers."""
    optimizer.zero_grad()
    if pgd_attack and adv_optimizer:
        adv_optimizer.zero_grad()


def _add_fedprox_regularization(loss_clean, model, wg_hat, mu):
    """Add FedProx regularization term to loss."""
    wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
    model_vec = parameters_to_vector(list(model.parameters()))
    prox_term = torch.norm(wg_hat_vec - model_vec) ** 2
    return loss_clean + mu / 2 * prox_term


def _perform_clean_training(model, optimizer, loss_clean, mask_grad_list, pgd_attack, proj,
                          model_original, adv_optimizer, batch_idx, total_batches, 
                          project_frequency, pgd_eps):
    """Perform clean training step."""
    optimizer.zero_grad()
    loss_clean.backward()
    
    if mask_grad_list:
        apply_grad_mask(model, mask_grad_list)
    
    if not pgd_attack:
        optimizer.step()
    else:
        _apply_pgd_projection(model, model_original, adv_optimizer, proj, batch_idx,
                            total_batches, project_frequency, pgd_eps)


def _perform_attack_training(model, atkmodel, tgtmodel, optimizer, adv_optimizer,
                           clean_images, clean_targets, poison_images, poison_targets,
                           output, loss_clean, attack_alpha, attack_portion, atk_eps,
                           clip_image, target_transform, pgd_attack, proj, model_original,
                           batch_idx, total_batches, project_frequency, pgd_eps,
                           mask_grad_list, bs, device, metrics):
    """Perform attack training step."""
    # Generate poisoned data
    with torch.no_grad():
        noise = tgtmodel(poison_images) * atk_eps
        atkdata = clip_image(poison_images + noise)
        atktarget = target_transform(poison_targets)
        
        if attack_portion < 1.0:
            atkdata = atkdata[:int(attack_portion * bs)]
            atktarget = atktarget[:int(attack_portion * bs)]
    
    # Forward pass for poisoned data
    atkoutput = model(atkdata.detach())
    loss_poison = F.cross_entropy(atkoutput, atktarget.detach())
    
    # Combined loss
    loss_combined = loss_clean * attack_alpha + (1.0 - attack_alpha) * loss_poison
    print(f"loss_clean: {loss_clean}, loss_poison: {loss_poison}")
    
    # Backward pass
    optimizer.zero_grad()
    loss_combined.backward()
    
    if mask_grad_list:
        apply_grad_mask(model, mask_grad_list)
    
    if not pgd_attack:
        optimizer.step()
    else:
        _apply_pgd_projection(model, model_original, adv_optimizer, proj, batch_idx,
                            total_batches, project_frequency, pgd_eps)
    
    # Update metrics
    metrics['loss_list'].append(loss_combined.item())
    pred = output.data.max(1)[1]
    poison_pred = atkoutput.data.max(1)[1]
    
    metrics['correct_clean'] += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()
    metrics['correct_poison'] += poison_pred.eq(atktarget.data.view_as(poison_pred)).cpu().sum().item()


def _apply_pgd_projection(model, model_original, adv_optimizer, proj, batch_idx,
                         total_batches, project_frequency, pgd_eps):
    """Apply PGD projection based on projection type."""
    if proj == "l_inf":
        _apply_l_inf_projection(model, model_original, pgd_eps)
    else:
        _apply_l2_projection(model, model_original, adv_optimizer, batch_idx,
                           total_batches, project_frequency, pgd_eps)


def _calculate_accuracy(correct, total):
    """Calculate accuracy percentage."""
    return 100.0 * (float(correct) / float(total)) if total else 0.0


def _log_training_results(training_avg_loss, clean_acc, poison_acc, atkmodel_train):
    """Log training results."""
    if atkmodel_train:
        logger.info(colored(f"Training loss = {training_avg_loss:.2f}, acc = {poison_acc:.2f} of atk model this epoch", "yellow"))
    else:
        logger.info(colored(f"Training loss = {training_avg_loss:.2f}, acc = {clean_acc:.2f} of cls model this epoch", "yellow"))
        logger.info(f"Training clean_acc is {clean_acc:.2f}, poison_acc = {poison_acc:.2f}")


def train_baseline(model, optimizer, train_loader, criterion, dataset, device, target_transform=None, local_e=0):
    """Train the baseline backdoored model on poisoned data.
    
    Args:
        model: Model to train
        optimizer: Optimizer for training
        train_loader: Data loader for training
        criterion: Loss function
        dataset: Dataset name
        device: Device to train on
        target_transform: Target transformation function
        local_e: Local epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    for batch_idx, batch in enumerate(train_loader):
        batch_size = len(batch)
        data, targets, poison_num, _, _ = get_poison_batch(
            batch, dataset, device, target_transform=target_transform
        )
        
        optimizer.zero_grad()
        dataset_size += len(data)
        poison_data_count += poison_num

        # Forward pass
        output = model(data)
        loss = nn.functional.cross_entropy(output, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()
    
    # Calculate final metrics
    avg_loss = total_loss / dataset_size
    accuracy = 100.0 * correct / dataset_size
    
    logger.info(f"Total loss is {avg_loss:.2f}, training acc is {accuracy:.2f}")
    return avg_loss, accuracy

def test(model, device, test_loader, test_batch_size, criterion, mode="raw-task", dataset="cifar10", poison_type="fashion", 
         fl_round=0, defense_method="no-defense", attack_method="baseline"):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    if dataset == 'tiny-imagenet':
        class_correct = list(0. for i in range(200))
        class_total = list(0. for i in range(200))
    
    if dataset in ("mnist", "emnist"):
        target_class = 7
        if mode == "raw-task":
            classes = [str(i) for i in range(10)]
        elif mode == "targetted-task":
            if poison_type == 'ardis':
                classes = [str(i) for i in range(10)]
            else: 
                classes = ["T-shirt/top", 
                            "Trouser",
                            "Pullover",
                            "Dress",
                            "Coat",
                            "Sandal",
                            "Shirt",
                            "Sneaker",
                            "Bag",
                            "Ankle boot"]
    
    elif dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # target_class = 2 for greencar, 9 for southwest
        if poison_type in ("howto", "greencar-neo"):
            target_class = 2
        else:
            target_class = 9
    
    elif dataset == "tiny-imagenet":
        classes = [str(i) for i in range(200)]
        target_class = 1
      
    model.eval()
    test_loss = 0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            if poison_type == 'ardis':
                backdoor_index = torch.where(target == target_class)
                target_backdoor = torch.ones_like(target[backdoor_index])
                predicted_backdoor = predicted[backdoor_index]
                backdoor_correct += (predicted_backdoor == target_backdoor).sum().item()
                backdoor_tot = backdoor_index[0].shape[0]
                
            #for image_index in range(test_batch_size):
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
        
                    
                class_total[label] += 1
    test_loss /= len(test_loader.dataset)
    
    number_class = 10
    if dataset == 'tiny-imagenet':
        number_class = 200
        
    if mode == "raw-task":
        for i in range(number_class):
            # logger.info('Accuracy of %5s : %.2f %%' % (
            #     classes[i], 100 * class_correct[i] / class_total[i]))

            if i == target_class:
                task_acc = 100 * class_correct[i] / class_total[i]

        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        final_acc = 100. * correct / len(test_loader.dataset)

    elif mode == "targetted-task":

        if dataset in ("mnist", "emnist"):
            for i in range(10):
                logger.info('Accuracy of %5s : %.2f %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
            if poison_type == 'ardis':
                # ensure 7 is being classified as 1
                logger.info('Backdoor Accuracy of %.2f : %.2f %%' % (
                     target_class, 100 * backdoor_correct / backdoor_tot))
                final_acc = 100 * backdoor_correct / backdoor_tot
            else:
                # trouser acc
                final_acc = 100 * class_correct[1] / class_total[1]
        
        elif dataset == "cifar10":
            logger.info('#### Targetted Accuracy of %5s : %.2f %%' % (classes[target_class], 100 * class_correct[target_class] / class_total[target_class]))
            final_acc = 100 * class_correct[target_class] / class_total[target_class]
    
    # Write test result to CSV
    csv_record.write_realtime_test_result(
        test_type=f"{mode}-task",
        fl_round=fl_round,
        model_identifier="baseline_model",
        clean_loss=test_loss,
        poison_loss=0.0,  # No poison loss for baseline test
        clean_accuracy=final_acc / 100.0,  # Convert percentage to decimal
        poison_accuracy=0.0,  # No poison accuracy for baseline test
        test_eps=0.0,  # No epsilon for baseline test
        dataset=dataset,
        defense_method=defense_method,
        attack_method=attack_method
    )
    
    return final_acc, task_acc

def iba_test(args, device, atkmodel, model, target_transform, 
         train_loader, test_loader, epoch, trainepoch, clip_image, 
         testoptimizer=None, log_prefix='Internal', epochs_per_test=3, 
         dataset="cifar10", criterion=None, subpath_saved="", test_eps=None, 
         is_poison=False, ident="global", atk_baseline=False, defense_method="no-defense", 
         attack_method="iba"):

    if args['test_alpha'] is None:
        args['test_alpha'] = args['attack_alpha']
    if args['test_eps'] is None:
        args['test_eps'] = args['eps']
    if not test_eps:
        test_eps = args['test_eps']
    
    model.eval()
    atkmodel.eval()
    test_loss = 0
    correct = 0
    correct_transform = 0
    test_loss = 0
    test_transform_loss = 0
    
    model = model.to(device)
    atkmodel = atkmodel.to(device)
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            start_time = time.time()
            bs = data.size(0)
            data, target = data.to(device), target.to(device)
            # batch = (data, target)
            output = model(data)
            
            test_loss += criterion(output, target).item() * bs  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print("Time for clean image: ", idx, time.time() - start_time)
            noise = atkmodel(data) * test_eps
            atkdata = clip_image(data + noise)
            atktarget = target_transform(target)
            # print("Time for atk image: ", idx, time.time() - start_time)
            
            atkoutput = model(atkdata)
            test_transform_loss += criterion(atkoutput, atktarget).item() * bs  # sum up batch loss
            atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_transform += atkpred.eq(
                target_transform(target).view_as(atkpred)).sum().item()

    # print(f"Finish the post testing------------------------------------")
    # print(f"Time for post testing: {time.time() - start_time}")
    
    test_loss /= len(test_loader.dataset)
    test_transform_loss /= len(test_loader.dataset)

    correct /= len(test_loader.dataset)
    correct_transform /= len(test_loader.dataset)
    # print(f"\nTest result without retraining: ")
    print(
        '{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
            log_prefix, 0, 
            test_loss, test_transform_loss,
            correct, correct_transform
        ))
    
    # Write real-time test result to CSV
    csv_record.write_realtime_test_result(
        test_type=log_prefix,
        fl_round=epoch,
        model_identifier=ident,
        clean_loss=test_loss,
        poison_loss=test_transform_loss,
        clean_accuracy=correct,
        poison_accuracy=correct_transform,
        test_eps=test_eps,
        dataset=dataset,
        defense_method=defense_method,
        attack_method=attack_method
    )
    
    if is_poison:
        csv_record.test_result.append([ident, epoch, test_loss, test_transform_loss, correct, correct_transform])
        # csv_record.save_result_csv(epoch, is_poison, subpath_saved)
    print(f"Time for end testing: {time.time() - start_time}")
    return correct, correct_transform
    
class FederatedLearningTrainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class BaseIBATrainer(FederatedLearningTrainer):
    """Base class for IBA-based federated learning trainers."""
    
    def __init__(self, arguments=None, iba_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_from_arguments(arguments, iba_args)
        self._setup_attack_configuration()
        self._setup_defense_mechanism()
        self._initialize_training_state()

    def _initialize_from_arguments(self, arguments, iba_args):
        """Initialize basic configuration from arguments."""
        # Validate arguments
        if arguments is None:
            raise ValueError("arguments cannot be None")
        if iba_args is None:
            raise ValueError("iba_args cannot be None")
        
        # Core FL parameters
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        
        # Models and data
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        
        # Training configuration
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.device = arguments['device']
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.criterion = nn.CrossEntropyLoss()
        
        # Attack configuration
        self.attacking_fl_rounds = arguments['attacking_fl_rounds']
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.attack_method = arguments["attack_method"]
        self.attack_case = arguments["attack_case"]
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.baseline = arguments['baseline']
        # self.retrain = arguments['retrain']
        self.retrain = True
        self.atk_baseline = arguments['atk_baseline']
        
        # Attack models
        self.atkmodel = arguments['atkmodel']
        self.tgtmodel = arguments['tgtmodel']
        self.scratch_model = arguments['scratch_model']
        
        # Attack parameters
        self.atk_eps = arguments['atk_eps']
        self.atk_test_eps = arguments['atk_test_eps']
        self.attack_alpha = arguments.get('attack_alpha', 0.5)
        self.attack_portion = arguments.get('attack_portion', 1.0)
        self.atk_lr = arguments['atk_lr']
        
        # Defense configuration
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.aggregator = arguments['aggregator']
        
        # PGD configuration
        self.eps = arguments['eps']
        self.eps_decay = arguments['eps_decay']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.prox_attack = arguments['prox_attack']
        self.stddev = arguments['stddev']
        
        # Scaling and other parameters
        self.scale_weights_poison = arguments['scale_weights_poison']
        self.scale_factor = arguments['scale']
        self.target_label = arguments.get('target_label', 1)
        self.save_model = arguments['save_model']
        self.folder_path = arguments['folder_path']
        
        # IBA specific
        self.iba_args = iba_args
        self.cur_training_eps = self.atk_eps

    def _setup_attack_configuration(self):
        """Setup attack method configuration."""
        if self.attack_method == "pgd":
            self.pgd_attack = True
            self.neurotoxin_attack = False
        elif self.attack_method == "neurotoxin":
            self.neurotoxin_attack = True
            self.pgd_attack = False
        else:
            self.pgd_attack = False
            self.neurotoxin_attack = False

    def _setup_defense_mechanism(self):
        """Setup defense mechanism based on configuration."""
        if self.defense_technique == "no-defense":
            self._defender = None
        elif self.defense_technique in ["norm-clipping", "norm-clipping-adaptive", "weak-dp"]:
            self._defender = WeightDiffClippingDefense(norm_bound=self.norm_bound)
        elif self.defense_technique in ["krum", "multi-krum"]:
            mode = self.defense_technique
            self._defender = Krum(mode=mode, num_workers=self.part_nets_per_round, num_adv=1)
        elif self.defense_technique == "rfa":
            self._defender = RFA()
        elif self.defense_technique == "crfl":
            self._defender = CRFL()
        elif self.defense_technique == "rlr":
            self._defender = self._create_rlr_defender()
        elif self.defense_technique == "foolsgold":
            self._defender = self._create_foolsgold_defender()
        else:
            raise NotImplementedError(f"Unsupported defense method: {self.defense_technique}")

    def _create_rlr_defender(self):
        """Create RLR defender with appropriate parameters."""
        pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
        args_rlr = {
            'aggr': 'avg',
            'noise': 0,
            'clip': 0,
            'server_lr': self.args_lr,
        }
        theta = 1
        return RLR(n_params=pytorch_total_params, device=self.device, 
                  args=args_rlr, robustLR_threshold=theta)

    def _create_foolsgold_defender(self):
        """Create FoolsGold defender with appropriate parameters."""
        pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
        return FoolsGold(num_clients=self.part_nets_per_round, 
                        num_classes=10, num_features=pytorch_total_params)

    def _initialize_training_state(self):
        """Initialize training state variables."""
        self.flatten_net_avg = None
        self.historical_grad_mask = None
        self.started_poisoning = False # Whether the poisoning phase (2nd phase) has started
        self.cnt_masks = 0
        self.mask_ratio = 0.30
        self.alternative_training = False # Whether to use alternative training to prevent catastrophic forgetting | only used in 2nd phase

    def _setup_training_environment(self):
        """Setup training environment and variables."""
        self.expected_atk_threshold = 0.85 #TODO: can be changed
        self.cnt_masks = 0
        
        # Initialize tracking lists
        self.main_task_acc = []
        self.raw_task_acc = []
        self.backdoor_task_acc = []
        self.fl_iter_list = []
        self.adv_norm_diff_list = []
        self.wg_norm_list = []
        
        # Initialize IBA variables
        self.trainlosses = []
        self.best_acc_clean = 0
        self.best_acc_poison = 0
        self.clip_image = get_clip_image(self.dataset)
        
        # Safety check for iba_args
        if self.iba_args is None:
            raise ValueError("iba_args cannot be None. Please ensure iba_args is properly initialized.")
        
        self.target_transform = get_target_transform(self.iba_args)
        
        # Setup paths and model loading
        self._setup_paths_and_models()
        
        # Initialize optimizers
        self.atk_optimizer = optim.Adam(self.atkmodel.parameters(), lr=self.atk_lr)
        
        # Initialize real-time CSV writer
        csv_filename = csv_record.initialize_realtime_csv_writer(
            output_dir="results", 
            filename_prefix=f"test_results_{self.dataset}_{self.attack_method}_{self.defense_technique}"
        )
        print(colored(f'Initialized real-time CSV writer: {csv_filename}', 'green'))
        
        print(colored(f'Under defense technique = {self.defense_technique}', 'red'))

    def _setup_paths_and_models(self):
        """Setup paths and load models if needed."""
        self.basepath, self.checkpoint_path, self.bestmodel_path = create_paths(self.iba_args)
        
        print('========== PATHS ==========')
        print(f'Basepath: {self.basepath}')
        print(f'Checkpoint Model: {self.checkpoint_path}')
        print(f'Best Model: {self.bestmodel_path}')
        
        # Load attack model if exists
        if os.path.exists(self.bestmodel_path) and not self.retrain:
            self._load_existing_attack_model()
        else:
            self._initialize_new_attack_model()

    def _load_existing_attack_model(self):
        """Load existing attack model from checkpoint."""
        checkpoint = torch.load(self.bestmodel_path)
        print(colored(f'Load existing attack model from path {self.bestmodel_path}', 'red'))
        self.atkmodel.load_state_dict(checkpoint['atkmodel'], strict=True)
        self.atk_model_loaded = True

    def _initialize_new_attack_model(self):
        """Initialize new attack model."""
        print(colored(f'Create new model from {self.bestmodel_path}', 'blue'))
        self.best_acc_clean = 0
        self.best_acc_poison = 0

    def _test_global_model_before_training(self):
        """Test the global model before starting federated training."""
        subpath_trigger_saved = self._get_subpath_trigger_saved()
        
        acc_clean, backdoor_acc = iba_test(
            self.iba_args, self.device, self.tgtmodel, self.net_avg, 
            self.target_transform, self.clean_train_loader, 
            self.vanilla_emnist_test_loader, 0, self.iba_args['train_epoch'], 
            self.clip_image, testoptimizer=None, log_prefix='External', 
            epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, 
            subpath_saved=subpath_trigger_saved, is_poison=True, 
            atk_baseline=self.atk_baseline, defense_method=self.defense_technique,
            attack_method=self.attack_method
        )
        
        print(f"\n----------TEST FOR GLOBAL MODEL BEFORE FEDERATED TRAINING----------------")
        print(f"Main task acc: {round(acc_clean*100, 2)}%")
        print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
        print(f"--------------------------------------------------------------------------\n")

    def _get_subpath_trigger_saved(self):
        """Get subpath for trigger model saving."""
        attack_train_epoch = self.iba_args['train_epoch']
        subpath = f"{self.dataset}baseline_{self.baseline}_atkepc_{attack_train_epoch}_eps_{self.atk_eps}_test_eps_{self.atk_test_eps}"
        return "" if self.baseline else subpath

    def _update_training_parameters(self, flr):
        """Update training parameters for current round."""
        if hasattr(self, 'local_acc_poison') and self.local_acc_poison >= self.expected_atk_threshold and not self.started_poisoning:
            # Start the poisoning phase (2nd phase)
            self.retrain = False # stop the generator training phase
            print(colored(f'Starting sub-training phase from flr = {flr}', 'red'))
            self.iba_args["test_eps"] = self.atk_test_eps
            self.start_decay_r = flr
            self.started_poisoning = True
        
        if self.started_poisoning:
            # Update the eps for the poisoning phase (2nd phase)
            self.cur_training_eps = max(
                self.atk_test_eps, 
                exponential_decay(self.atk_eps, self.eps_decay, flr - getattr(self, 'start_decay_r', 0))
            )
            # Toggle the alternative training flag | only used in 2nd phase \ avoid catastrophic forgetting
            self.alternative_training = not self.alternative_training

    def _train_attacker(self, net, train_dl_local, flr, num_data_points, total_num_dps_per_round):
        """Train the attacker client."""
        # Setup optimizers
        optimizer = self._create_optimizer(net, flr)
        
        # Training parameters
        attack_portion = 0.0 if not self.started_poisoning else self.iba_args['attack_portion']
        atk_alpha = self.iba_args['attack_alpha'] if self.started_poisoning else 1.0
        pgd_attack = False if self.cur_training_eps == self.atk_eps else self.pgd_attack
        
        print(colored(f"At flr {flr}, test eps is {self.cur_training_eps}", "green"))
        
        # Adversarial training
        for e in range(1, self.adversarial_local_training_period + 1):
            if not self.atk_baseline:
                self._train_attacker_advanced(net, train_dl_local, optimizer, None, 
                                            e, attack_portion, atk_alpha, pgd_attack, flr)
            else:
                train_baseline(net, optimizer, train_dl_local, self.criterion, 
                             self.dataset, self.device, self.target_transform, e)
        
        # Retrain attack model if needed
        if (self.retrain or self.alternative_training) and not self.atk_baseline:
            self.tgtmodel.eval()
            self.atkmodel.load_state_dict(self.tgtmodel.state_dict())
            adv_optimizer = self._create_adv_optimizer(self.atkmodel, flr)
            self._retrain_attack_model(net, train_dl_local, self.atkmodel, adv_optimizer)
            self.tgtmodel.load_state_dict(self.atkmodel.state_dict())
        
        # Apply scaling if needed
        self._apply_attacker_scaling(net, total_num_dps_per_round, flr)
        
        # Test attacker performance
        self._test_attacker_performance(net, e, flr)

    def _train_attacker_advanced(self, net, train_dl_local, optimizer, adv_optimizer, 
                               epoch, attack_portion, atk_alpha, pgd_attack, flr):
        """Advanced attacker training with IBA."""
        pdg_eps = self.eps * self.args_gamma**(flr - getattr(self, 'start_decay_r', 0)) if self.defense_technique in ('krum', 'multi-krum') else self.eps
        mask_grad_list = []
        if self.started_poisoning and self.neurotoxin_attack:
            self.cnt_masks += 1
            clone_local_model = copy.deepcopy(net)
            clone_optimizer = optim.SGD(clone_local_model.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
            mask_grad_list, self.historical_grad_mask = get_grad_mask(clone_local_model, clone_optimizer, train_dl_local, self.mask_ratio, self.device, self.historical_grad_mask, cnt_masks=self.cnt_masks)
        
        train_iba_local(net, self.atkmodel, self.tgtmodel, optimizer, None, 
                       train_dl_local, self.criterion, self.cur_training_eps, atk_alpha, 
                       attack_portion, self.clip_image, self.target_transform, 1, 
                       atkmodel_train=False, device=self.device, pgd_attack=pgd_attack, 
                       batch_idx=epoch, project_frequency=self.project_frequency,
                       pgd_eps=pdg_eps, model_original=copy.deepcopy(self.net_avg), 
                       adv_optimizer=None, mask_grad_list=mask_grad_list, 
                       aggregator=self.aggregator, wg_hat=copy.deepcopy(self.net_avg), 
                       local_e=epoch)

    def _retrain_attack_model(self, net, train_dl_local, atk_model, atk_optimizer):
        
        """Update the attack model to avoid catastrophic forgetting."""
        # This is only used in the 1st phase (generator training phase) - self.retrain = True
        # This could be used in the 2nd phase (poisoning phase) - self.retrain = False but self.alternative_training = True
        attack_train_epoch = self.iba_args['train_epoch']
        lira_train_eps = self.cur_training_eps
        for e in range(1, attack_train_epoch + 1):
            train_iba_local(net, atk_model, self.tgtmodel, None, atk_optimizer, 
                           train_dl_local, self.criterion, lira_train_eps, self.attack_alpha, 
                           1.0, self.clip_image, self.target_transform, 1, 
                           atkmodel_train=True, device=self.device, pgd_attack=False)

    def _apply_attacker_scaling(self, net, total_num_dps_per_round, flr):
        """Apply scaling to attacker model if needed."""
        model_original = list(self.net_avg.parameters())
        
        # Model replacement scaling
        replacement_flr = 400 if self.dataset == "mnist" else 500
        if self.model_replacement and self.cur_training_eps != self.atk_eps and flr >= replacement_flr:
            v = torch.nn.utils.parameters_to_vector(net.parameters())
            logger.info(f"Attacker before scaling : Norm = {torch.norm(v)}")
            
            for idx, param in enumerate(net.parameters()):
                param.data = (param.data - model_original[idx]) * (total_num_dps_per_round / self.num_dps_poisoned_dataset) + model_original[idx]
            
            v = torch.nn.utils.parameters_to_vector(net.parameters())
            logger.info(f"Attacker after scaling : Norm = {torch.norm(v)}")
        
        # Weight scaling
        if self.scale_weights_poison != 1.0 and flr % 501 == 0:
            v = torch.nn.utils.parameters_to_vector(net.parameters())
            logger.info(f"Attacker before scaling with factor {self.scale_weights_poison}: Norm = {torch.norm(v)}")
            
            for idx, param in enumerate(net.parameters()):
                param.data = (param.data - model_original[idx]) * self.scale_weights_poison + model_original[idx]
            
            v = torch.nn.utils.parameters_to_vector(net.parameters())
            logger.info(f"Attacker after scaling : Norm = {torch.norm(v)}")

    def _test_attacker_performance(self, net, epoch, flr):
        """Test attacker performance."""
        subpath_trigger_saved = self._get_subpath_trigger_saved()
        
        self.local_acc_clean, self.local_acc_poison = iba_test(
            self.iba_args, self.device, self.tgtmodel, net, self.target_transform, 
            self.clean_train_loader, self.vanilla_emnist_test_loader, flr, 
            self.iba_args['train_epoch'], self.clip_image, log_prefix='Internal', 
            dataset=self.dataset, criterion=self.criterion, 
            subpath_saved=subpath_trigger_saved, test_eps=self.cur_training_eps, 
            is_poison=True, atk_baseline=self.atk_baseline, defense_method=self.defense_technique,
            attack_method=self.attack_method
        )
        print(f"[DEBUG]: local acc clean: {self.local_acc_clean}, local acc poison: {self.local_acc_poison}")

    def _train_benign_client(self, net_idx, net, global_user_idx, flr):
        """Train a benign client."""
        dataidxs = self.net_dataidx_map[global_user_idx]
        train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                         self.test_batch_size, dataidxs)
        logger.info(f"@@@@@@@@ Working on client: {net_idx}, which is Global user: {global_user_idx}")
        self._train_benign_client_internal(net, train_dl_local, flr)

    def _train_benign_client_internal(self, net, train_dl_local, flr):
        """Internal training for benign client."""
        optimizer = self._create_optimizer(net, flr)
        
        for e in range(1, self.local_training_period + 1):
            train_iba_local(net, self.atkmodel, self.tgtmodel, optimizer, self.atk_optimizer, 
                          train_dl_local, self.criterion, self.cur_training_eps, 1.0, 0.0, 
                          self.clip_image, self.target_transform, 1, atkmodel_train=False, 
                          device=self.device, aggregator=self.aggregator, 
                          wg_hat=copy.deepcopy(self.net_avg))

    def _create_optimizer(self, net, flr):
        """Create optimizer for client training."""
        optimizer = optim.SGD(net.parameters(), lr=self.args_lr * self.args_gamma**(flr-1), 
                            momentum=0.9, weight_decay=1e-4)
        for param_group in optimizer.param_groups:
            logger.info(f"Effective lr in FL round: {flr} is {param_group['lr']}")
        return optimizer

    def _create_adv_optimizer(self, net, flr):
        """Create adversarial optimizer."""
        return optim.SGD(net.parameters(), lr=self.atk_lr, 
                        momentum=0.9, weight_decay=1e-4)

    def _apply_defense_mechanisms(self):
        """Apply defense mechanisms to client models."""
        print(colored(f"Conduct defense technique: {self.defense_technique}", "red"))
        
        if self.defense_technique == "no-defense":
            pass
        elif self.defense_technique in ["norm-clipping", "weak-dp"]:
            for net in self.current_net_list:
                self._defender.exec(client_model=net, global_model=self.net_avg)
        elif self.defense_technique == "norm-clipping-adaptive":
            self._apply_adaptive_norm_clipping()
        elif self.defense_technique in ["krum", "multi-krum"]:
            self._apply_krum_defense()
        elif self.defense_technique == "foolsgold":
            self._apply_foolsgold_defense()
        elif self.defense_technique in ["rfa", "crfl", "rlr"]:
            self._apply_other_defenses()

    def _apply_adaptive_norm_clipping(self):
        """Apply adaptive norm clipping defense."""
        # This would need to be implemented based on the norm_diff_collector logic
        pass

    def _apply_krum_defense(self):
        """Apply Krum or Multi-Krum defense."""
        num_dps = [self.num_dps_poisoned_dataset] + [len(self.net_dataidx_map[i]) for i in self.current_selected_indices]
        self.current_net_list, self.current_net_freq = self._defender.exec(
            client_models=self.current_net_list, 
            num_dps=num_dps,
            g_user_indices=self.current_selected_indices,
            device=self.device
        )

    def _apply_foolsgold_defense(self):
        """Apply FoolsGold defense."""
        # This would need to be implemented based on the FoolsGold logic
        pass

    def _apply_other_defenses(self):
        """Apply other defense mechanisms."""
        # This would need to be implemented for RFA, CRFL, RLR
        pass

    def _aggregate_models(self):
        """Aggregate client models using the specified aggregator."""
        fed_aggregator = self.aggregator
        print(f"\nStart performing fed aggregator: {fed_aggregator}...")
        
        if fed_aggregator == "fedavg":
            self.net_avg = fed_avg_aggregator(self.net_avg, self.current_net_list, 
                                            self.current_net_freq, device=self.device, 
                                            model=self.model)
        elif fed_aggregator == "fednova":
            # This would need to be implemented with proper FedNova parameters
            pass
        else:
            raise NotImplementedError(f"Unsupported fed agg {fed_aggregator} method !")

    def _test_and_log_results(self, flr, wandb_ins):
        """Test the aggregated model and log results."""
        subpath_trigger_saved = self._get_subpath_trigger_saved()
        
        acc_clean, backdoor_acc = iba_test(
            args=self.iba_args, device=self.device, atkmodel=self.tgtmodel, 
            model=self.net_avg, target_transform=self.target_transform, 
            train_loader=self.clean_train_loader, test_loader=self.vanilla_emnist_test_loader,
            epoch=flr, trainepoch=self.iba_args['train_epoch'], clip_image=self.clip_image, 
            testoptimizer=None, log_prefix='External', epochs_per_test=3, 
            dataset=self.dataset, criterion=self.criterion, 
            subpath_saved=subpath_trigger_saved, test_eps=self.cur_training_eps,
            is_poison=True, atk_baseline=self.atk_baseline, defense_method=self.defense_technique,
            attack_method=self.attack_method
        )
        
        # Apply weak-DP noise if needed
        if self.defense_technique == "weak-dp":
            noise_adder = AddNoise(stddev=self.stddev)
            noise_adder.exec(client_model=self.net_avg, device=self.device)
        
        # Log results
        self._log_round_results(flr, acc_clean, backdoor_acc, wandb_ins)
        
        # Save model if needed
        self._save_model_if_needed(flr)

    def _log_round_results(self, flr, acc_clean, backdoor_acc, wandb_ins):
        """Log results for the current round."""
        v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
        logger.info(f"############ Averaged Model : Norm {torch.norm(v)}")
        
        calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.vanilla_model, 
                      epoch=0, fl_round=flr, mode="avg")
        
        logger.info(f"Measuring the accuracy of the averaged global model, FL round: {flr} ...")
        
        # Update tracking lists
        self.fl_iter_list.append(flr)
        self.main_task_acc.append(acc_clean)
        self.raw_task_acc.append(0)  # This seems to be always 0
        self.backdoor_task_acc.append(backdoor_acc)
        
        # Log to wandb if available
        if wandb_ins:
            wandb_logging_items = {
                'fl_iter': flr,
                'main_task_acc': acc_clean * 100.0,
                'backdoor_task_acc': backdoor_acc * 100.0,
                'local_best_acc_clean': self.best_acc_clean,
                'local_best_acc_poison': self.best_acc_poison,
                'local_MA': getattr(self, 'local_acc_clean', 0) * 100.0,
                'local_BA': getattr(self, 'local_acc_poison', 0) * 100.0,
                'cur_training_eps': self.cur_training_eps,
            }
            wandb_ins.log({"General Information": wandb_logging_items})

    def _save_model_if_needed(self, flr):
        """Save model if conditions are met."""
        if flr >= 200 and flr % 50 == 0 and self.save_model:
            current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
            folder_path = f'saved_models/{self.folder_path}_{"baseline" if self.baseline else "poison"}_{current_time}'
            
            try:
                os.makedirs(folder_path)
            except FileExistsError:
                logger.info('Folder already exists')
            
            model_name = f'{folder_path}/model_last_flr_{flr}.pt'
            atk_model_name = f'{folder_path}/atkmodel_last_flr_{flr}.pt'
            
            torch.save(self.net_avg.state_dict(), model_name)
            torch.save(self.tgtmodel.state_dict(), atk_model_name)

    def _save_final_results(self):
        """Save final results and models."""
        # Save final model
        saved_path = "checkpoint/last_model"
        torch.save(self.net_avg.state_dict(), 
                  f"{saved_path}/lira_{self.dataset}_{self.model}_{self.atk_test_eps}.pt")
        
        # Create results DataFrame
        df = pd.DataFrame({
            'fl_iter': self.fl_iter_list, 
            'main_task_acc': self.main_task_acc, 
            'backdoor_acc': self.backdoor_task_acc, 
            'raw_task_acc': self.raw_task_acc, 
            'adv_norm_diff': self.adv_norm_diff_list, 
            'wg_norm': self.wg_norm_list
        })
        
        # Add initial accuracies for ARDIS dataset
        if self.poison_type == 'ardis':
            df1 = pd.DataFrame({
                'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 
                'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]
            })
            df = pd.concat([df1, df])
        
        # Save results
        results_filename = get_results_filename(
            self.poison_type, self.attack_method, self.model_replacement, 
            self.project_frequency, self.defense_technique, self.norm_bound, 
            self.prox_attack, False, self.model
        )
        
        df.to_csv(results_filename, index=False)
        logger.info(f"Wrote accuracy results to: {results_filename}")
        
        # Close real-time CSV writer and print summary
        csv_record.close_realtime_csv_writer()
        summary = csv_record.get_realtime_test_summary()
        if summary:
            logger.info(f"Real-time test summary: {summary}")


class FrequencyFederatedLearningTrainer(BaseIBATrainer):
    """Frequency-based federated learning trainer with IBA."""
    
    def __init__(self, arguments=None, iba_args=None, *args, **kwargs):
        super().__init__(arguments, iba_args, *args, **kwargs)
        
    def run(self, wandb_ins):
        """Main training loop for federated learning."""
        self._setup_training_environment()
        
        # Test global model before training
        self._test_global_model_before_training()
        
        # Main FL training loop
        for flr in range(1, self.fl_round + 1):
            self._execute_fl_round(flr, wandb_ins)
        
        # Save final results
        self._save_final_results()

    def _execute_fl_round(self, flr, wandb_ins):
        """Execute a single federated learning round."""
        print("---" * 30)
        print(f"Start communication round {flr}")
        
        # Update training parameters
        self._update_training_parameters(flr)
        
        # Select clients and prepare for training
        if flr in self.attacking_fl_rounds and not self.baseline:
            self._execute_attack_round(flr)
        else:
            self._execute_benign_round(flr)
        
        # Apply defense mechanisms
        self._apply_defense_mechanisms()
        
        # Aggregate models
        self._aggregate_models()
        
        # Test and log results
        self._test_and_log_results(flr, wandb_ins)

    def _execute_attack_round(self, flr):
        """Execute a federated learning round with attackers."""
        # Select participating clients
        selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
        num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
        total_num_dps_per_round = sum(num_data_points)
        
        # Prepare network list and frequencies
        net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
        net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
        
        logger.info(f"Net freq: {net_freq}, FL round: {flr} with adversary")
        
        # Train each client
        for net_idx, net in enumerate(net_list):
            self._train_client(net_idx, net, selected_node_indices[net_idx], 
                             num_data_points, total_num_dps_per_round, flr)
        
        # Store for aggregation
        self.current_net_list = net_list
        self.current_net_freq = net_freq
        self.current_selected_indices = selected_node_indices

    def _execute_benign_round(self, flr):
        """Execute a federated learning round without attackers."""
        selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
        num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
        total_num_dps_per_round = sum(num_data_points)
        
        net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
        logger.info(f"Net freq: {net_freq}, FL round: {flr} without adversary")
        
        net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
        
        # Train each client
        for net_idx, net in enumerate(net_list):
            self._train_benign_client(net_idx, net, selected_node_indices[net_idx], flr)
        
        # Store for aggregation
        self.current_net_list = net_list
        self.current_net_freq = net_freq
        self.current_selected_indices = selected_node_indices

    def _train_client(self, net_idx, net, global_user_idx, num_data_points, total_num_dps_per_round, flr):
        """Train a single client (attacker or benign)."""
        # Setup data loader
        if net_idx == 0:  # Attacker
            train_dl_local = copy.deepcopy(self.clean_train_loader)
            logger.info(f"@@@@@@@@ Working on client: {net_idx}, which is Attacker")
            self._train_attacker(net, train_dl_local, flr, num_data_points, total_num_dps_per_round)
        else:  # Benign client
            dataidxs = self.net_dataidx_map[global_user_idx]
            train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                             self.test_batch_size, dataidxs)
            logger.info(f"@@@@@@@@ Working on client: {net_idx}, which is Global user: {global_user_idx}")
            self._train_benign_client_internal(net, train_dl_local, flr)


class FixedPoolFederatedLearningTrainer(BaseIBATrainer):
    """Fixed pool federated learning trainer with IBA."""
    
    def __init__(self, arguments=None, iba_args=None, *args, **kwargs):
        super().__init__(arguments, iba_args, *args, **kwargs)
        # Fixed pool specific initialization
        self.attacker_pool_size = arguments.get('attacker_pool_size', 0)
        self.__attacker_pool = np.random.choice(self.num_nets, self.attacker_pool_size, replace=False)

    def run(self, wandb_ins):
        """Main training loop for fixed pool federated learning."""
        self._setup_training_environment()
        
        # Test global model before training
        self._test_global_model_before_training()
        
        # Main FL training loop
        for flr in range(1, self.fl_round + 1):
            self._execute_fixed_pool_round(flr, wandb_ins)
        
        # Save final results
        self._save_final_results()

    def _execute_fixed_pool_round(self, flr, wandb_ins):
        """Execute a single federated learning round with fixed pool."""
        print("---" * 30)
        print(f"Start communication round {flr}")
        
        # Update training parameters
        self._update_training_parameters(flr)
        
        # Execute round based on whether it's an attack round
        self._execute_fixed_pool_attack_round(flr)
        
        # Apply defense mechanisms
        self._apply_defense_mechanisms()
        
        # Aggregate models
        self._aggregate_models()
        
        # Test and log results
        self._test_and_log_results(flr, wandb_ins)

    def _execute_fixed_pool_attack_round(self, flr):
        """Execute a federated learning round with fixed pool attackers."""
        # Select participating clients
        selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
        selected_attackers = [idx for idx in selected_node_indices if idx in self.__attacker_pool]
        
        logger.info("Selected Attackers in FL iteration-{}: {}".format(flr, selected_attackers))

        # Calculate data points for each selected client
        num_data_points = []
        for sni in selected_node_indices:
            if sni in selected_attackers:
                num_data_points.append(self.num_dps_poisoned_dataset)
            else:
                num_data_points.append(len(self.net_dataidx_map[sni]))

        total_num_dps_per_round = sum(num_data_points)
        net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
        logger.info("Net freq: {}, FL round: {} with fixed pool adversary".format(net_freq, flr))

        # Create network list
        net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
        atk_model_list = {g_id: copy.deepcopy(self.tgtmodel) for g_id in selected_attackers}
        logger.info("################## Starting fl round: {}".format(flr))
        
        # Train each client using the correct mapping
        for net_idx, net in enumerate(net_list):
            global_user_idx = selected_node_indices[net_idx]  # Get the actual client ID
            
            if global_user_idx in selected_attackers:
                # This is an attacker from the fixed pool
                self._train_fixed_pool_attacker(net_idx, net, global_user_idx, 
                                              num_data_points, total_num_dps_per_round, flr,
                                              atk_model_list[global_user_idx])
            else:
                # This is a benign client
                self._train_benign_client(net_idx, net, global_user_idx, flr)
        
        # Store for aggregation

        # aggregation of the attack models distributed among participants
        atk_model_freq = [1.0/len(selected_attackers) for _ in range(len(selected_attackers))]
        if len(selected_attackers) > 0:
            self.atkmodel = fed_avg_aggregator(self.atkmodel, list(atk_model_list.values()), atk_model_freq, device=self.device, model="atkmodel")
            self.tgtmodel.load_state_dict(self.atkmodel.state_dict())
        self.current_net_list = net_list
        self.current_net_freq = net_freq
        self.current_selected_indices = selected_node_indices

    def _execute_fixed_pool_benign_round(self, flr):
        """Execute a federated learning round without attackers (same as frequency)."""
        selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
        num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
        total_num_dps_per_round = sum(num_data_points)
        
        net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
        logger.info(f"Net freq: {net_freq}, FL round: {flr} without adversary")
        
        net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
        
        # Train each client
        for net_idx, net in enumerate(net_list):
            self._train_benign_client(net_idx, net, selected_node_indices[net_idx], flr)
        
        # Store for aggregation
        self.current_net_list = net_list
        self.current_net_freq = net_freq
        self.current_selected_indices = selected_node_indices

    def _train_fixed_pool_attacker(self, net_idx, net, global_user_idx, num_data_points, total_num_dps_per_round, flr, pool_attacker_model):
        """Train an attacker from the fixed pool."""
        # Use the corresponding attacker model from the pool using global_user_idx
        # pool_attacker_model = copy.deepcopy(self.atkmodel)
        
        # Setup data loader
        train_dl_local = copy.deepcopy(self.clean_train_loader)
        logger.info(f"@@@@@@@@ Working on client: {net_idx}, which is Fixed Pool Attacker (Global ID: {global_user_idx})")
        
        # Train using the pool attacker model
        self._train_attacker_with_model(net, pool_attacker_model, train_dl_local, flr, num_data_points, total_num_dps_per_round)

    def _train_attacker_with_model(self, net, attacker_model, train_dl_local, flr, num_data_points, total_num_dps_per_round):
        """Train attacker with a specific attacker model."""
        # Setup optimizers
        optimizer = self._create_optimizer(net, flr)
        adv_optimizer = self._create_adv_optimizer(attacker_model, flr)
        print(colored(f"[DEBUG]: adv_optimizer: {adv_optimizer}", "red"))
        # Training parameters
        attack_portion = 0.0 if self.retrain else self.iba_args['attack_portion']
        atk_alpha = self.iba_args['attack_alpha'] if not self.retrain else 1.0
        pgd_attack = False if self.cur_training_eps == self.atk_eps else self.pgd_attack
        
        print(colored(f"At flr {flr}, test eps is {self.cur_training_eps}", "green"))
        
        # Adversarial training with the specific attacker model
        for e in range(1, self.adversarial_local_training_period + 1):
            if not self.atk_baseline:
                self._train_attacker_advanced_with_model(net, attacker_model, train_dl_local, 
                                                       optimizer, adv_optimizer, e, attack_portion, 
                                                       atk_alpha, pgd_attack, flr)
            else:
                train_baseline(net, optimizer, train_dl_local, self.criterion, 
                             self.dataset, self.device, self.target_transform, e)
        
        # Apply scaling if needed
        self._apply_attacker_scaling(net, total_num_dps_per_round, flr)
        
        # Test attacker performance
        self._test_attacker_performance_with_model(net, attacker_model, e, flr)

    def _train_attacker_advanced_with_model(self, net, attacker_model, train_dl_local, optimizer, 
                                          adv_optimizer, epoch, attack_portion, atk_alpha, pgd_attack, flr):
        """Advanced attacker training with IBA using a specific attacker model."""
        pdg_eps = self.eps * self.args_gamma**(flr - getattr(self, 'start_decay_r', 0)) if self.defense_technique in ('krum', 'multi-krum') else self.eps
        mask_grad_list = []
        if self.started_poisoning and self.neurotoxin_attack:
            self.cnt_masks += 1
            clone_local_model = copy.deepcopy(net)
            clone_optimizer = optim.SGD(clone_local_model.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
            mask_grad_list, self.historical_grad_mask = get_grad_mask(clone_local_model, clone_optimizer, train_dl_local, self.mask_ratio, self.device, self.historical_grad_mask, cnt_masks=self.cnt_masks)
        
        train_iba_local(net, attacker_model, self.tgtmodel, optimizer, adv_optimizer, 
                       train_dl_local, self.criterion, self.cur_training_eps, atk_alpha, 
                       attack_portion, self.clip_image, self.target_transform, 1, 
                       atkmodel_train=False, device=self.device, pgd_attack=pgd_attack, 
                       batch_idx=epoch, project_frequency=self.project_frequency,
                       pgd_eps=pdg_eps, model_original=copy.deepcopy(self.net_avg), 
                       adv_optimizer=adv_optimizer, mask_grad_list=mask_grad_list, 
                       aggregator=self.aggregator, wg_hat=copy.deepcopy(self.net_avg), 
                       local_e=epoch)
        # Update the attack model if needed
        if (self.retrain or self.alternative_training) and not self.atk_baseline:
            self._retrain_attack_model(net, train_dl_local, attacker_model, adv_optimizer)

    def _test_attacker_performance_with_model(self, net, attacker_model, epoch, flr):
        """Test attacker performance with a specific attacker model."""
        subpath_trigger_saved = self._get_subpath_trigger_saved()
        
        self.local_acc_clean, self.local_acc_poison = iba_test(
            self.iba_args, self.device, attacker_model, net, self.target_transform, 
            self.clean_train_loader, self.vanilla_emnist_test_loader, flr, 
            self.iba_args['train_epoch'], self.clip_image, log_prefix='Internal', 
            dataset=self.dataset, criterion=self.criterion, 
            subpath_saved=subpath_trigger_saved, test_eps=self.cur_training_eps, 
            is_poison=True, atk_baseline=self.atk_baseline, defense_method=self.defense_technique,
            attack_method=self.attack_method
        )
