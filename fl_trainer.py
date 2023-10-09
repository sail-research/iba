import numpy as np

import torch
import torch.nn.functional as F
import datetime
import time
from utils import *
from defense import *

from tqdm.auto import tqdm

from models.vgg import get_vgg_model
import pandas as pd

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from lira_helper import *
import csv_record

import datasets
from termcolor import colored

def exponential_decay(init_val, decay_rate, t):
    return init_val*(1.0 - decay_rate)**t

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
  
def train_lira(model, atkmodel, tgtmodel, optimizer, atkmodel_optimizer, train_loader, criterion, atk_eps = 0.001, attack_alpha = 0.5,
               attack_portion = 1.0, clip_image=None, target_transform=None, atk_train_epoch=1, atkmodel_train=False, device=None, 
               pgd_attack = False, batch_idx = 1, project_frequency = 1, pgd_eps = None, model_original=None, adv_optimizer=None, 
               proj="l_2", mask_grad_list=[], aggregator="fedavg", wg_hat=None, mu=0.1, local_e=0):
    # print(f"attack_alpha: {attack_alpha}")
    # print(f"atkmodel_optimizer: {atkmodel_optimizer}")
    tgtmodel.eval()
    wg_clone = copy.deepcopy(model)
    # atk_copy = copy.deepcopy(tgtmodel)
    # atk_optimizer = None
    loss_fn = nn.CrossEntropyLoss()
    func_fn = loss_fn
    correct = 0
    
    correct_clean = 0
    correct_poison = 0
    
    dataset_size = 0
    poison_size = 0
    clean_size = 0
    loss_list = []
    
    if not atkmodel_train:
        # optimizer = optim.SGD(model.parameters(), lr=0.001)
        model.train()
        # atkmodel.eval()
        # Sub-training phase
        for batch_idx, batch in enumerate(train_loader):
            bs = len(batch)
            data, targets = batch
            # data, target = data.to(device), target.to(device)
            # clean_images, clean_targets, poison_images, poison_targets, poisoning_per_batch = get_poison_batch(batch, attack_portion)
            clean_images, clean_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
            poison_images, poison_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
            # dataset_size += len(data)   
            clean_size += len(clean_images)
            optimizer.zero_grad()
            if pgd_attack:
                adv_optimizer.zero_grad()
            output = model(clean_images)
            loss_clean = loss_fn(output, clean_targets)
            if aggregator == "fedprox":
                # print(f"fedprox!!!")
                wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
                model_vec = parameters_to_vector(list(model.parameters()))
                prox_term = torch.norm(wg_hat_vec - model_vec)**2
                loss_clean = loss_clean + mu/2*prox_term
            if attack_alpha == 1.0:
                optimizer.zero_grad()
                # atkmodel_optimizer.zero_grad()
                loss_clean.backward()
                if mask_grad_list:
                    apply_grad_mask(model, mask_grad_list)
                if not pgd_attack:
                    optimizer.step()
                else:
                    if proj == "l_inf":
                        w = list(model.parameters())
                        n_layers = len(w)
                        # adversarial learning rate
                        eta = 0.001
                        for i in range(len(w)):
                            # uncomment below line to restrict proj to some layers
                            if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                                w[i].data = w[i].data - eta * w[i].grad.data
                                # projection step
                                m1 = torch.lt(torch.sub(w[i], model_original[i]), -pgd_eps)
                                m2 = torch.gt(torch.sub(w[i], model_original[i]), pgd_eps)
                                w1 = (model_original[i] - pgd_eps) * m1
                                w2 = (model_original[i] + pgd_eps) * m2
                                w3 = (w[i]) * (~(m1+m2))
                                wf = w1+w2+w3
                                w[i].data = wf.data
                    else:
                        # do l2_projection
                        adv_optimizer.step()
                        w = list(model.parameters())
                        w_vec = parameters_to_vector(w)
                        model_original_vec = parameters_to_vector(model_original)
                        # make sure you project on last iteration otherwise, high LR pushes you really far
                        # Start
                        if (batch_idx%project_frequency == 0 or batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > pgd_eps):
                            # project back into norm ball
                            w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
                                    w_vec-model_original_vec) + model_original_vec
                            # plug w_proj back into model
                            vector_to_parameters(w_proj_vec, w)

                        
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                loss_list.append(loss_clean.item())
                correct_clean += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()
            else:
                if attack_alpha < 1.0:
                    poison_size += len(poison_images)
                    # poison_images, poison_targets = poison_images.to(device), poison_targets.to(device)
                    with torch.no_grad():
                        noise = tgtmodel(poison_images) * atk_eps
                        atkdata = clip_image(poison_images + noise)
                        atktarget = target_transform(poison_targets)
                        # atkdata.requires_grad_(False)
                        # atktarget.requires_grad_(False)
                        if attack_portion < 1.0:
                            atkdata = atkdata[:int(attack_portion*bs)]
                            atktarget = atktarget[:int(attack_portion*bs)]
                    # import IPython
                    # IPython.embed()
                    atkoutput = model(atkdata.detach())
                    loss_poison = F.cross_entropy(atkoutput, atktarget.detach())
                else:
                    loss_poison = torch.tensor(0.0).to(device)
                loss2 = loss_clean * attack_alpha  + (1.0 - attack_alpha) * loss_poison
                
                optimizer.zero_grad()
                loss2.backward()
                if mask_grad_list:
                    apply_grad_mask(model, mask_grad_list)
                if not pgd_attack:
                    optimizer.step()
                else:
                    if proj == "l_inf":
                        w = list(model.parameters())
                        n_layers = len(w)
                        # adversarial learning rate
                        eta = 0.001
                        for i in range(len(w)):
                            # uncomment below line to restrict proj to some layers
                            if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                                w[i].data = w[i].data - eta * w[i].grad.data
                                # projection step
                                m1 = torch.lt(torch.sub(w[i], model_original[i]), -pgd_eps)
                                m2 = torch.gt(torch.sub(w[i], model_original[i]), pgd_eps)
                                w1 = (model_original[i] - pgd_eps) * m1
                                w2 = (model_original[i] + pgd_eps) * m2
                                w3 = (w[i]) * (~(m1+m2))
                                wf = w1+w2+w3
                                w[i].data = wf.data
                    else:
                        # do l2_projection
                        adv_optimizer.step()
                        w = list(model.parameters())
                        w_vec = parameters_to_vector(w)
                        model_original_vec = parameters_to_vector(list(model_original.parameters()))
                        # make sure you project on last iteration otherwise, high LR pushes you really far
                        if (local_e%project_frequency == 0 and batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > pgd_eps):
                            # project back into norm ball
                            w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
                                    w_vec-model_original_vec) + model_original_vec
                            
                          
                            vector_to_parameters(w_proj_vec, w)

                loss_list.append(loss2.item())
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                poison_pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability
                
                # correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                correct_clean += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()
                correct_poison += poison_pred.eq(atktarget.data.view_as(poison_pred)).cpu().sum().item()
                
    else:
        model.eval()
        # atk_optimizer = optim.Adam(atkmodel.parameters(), lr=0.0002)
        atkmodel.train()
        # optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(train_loader):
            bs = data.size(0)
            data, target = data.to(device), target.to(device)
            # dataset_size += len(data)
            poison_size += len(data)
            
            ###############################
            #### Update the classifier ####
            ###############################
            # with torch.no_grad():
            noise = atkmodel(data) * atk_eps
            atkdata = clip_image(data + noise)
            atktarget = target_transform(target)
            if attack_portion < 1.0:
                atkdata = atkdata[:int(attack_portion*bs)]
                atktarget = atktarget[:int(attack_portion*bs)]     
            # with torch.no_grad():
            # atkoutput = wg_clone(atkdata)
            atkoutput = model(atkdata)
            loss_p = func_fn(atkoutput, atktarget)
            loss2 = loss_p
            # import IPython
            # IPython.embed()
            atkmodel_optimizer.zero_grad()
            loss2.backward()
            atkmodel_optimizer.step()
            pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability
            correct_poison += pred.eq(atktarget.data.view_as(pred)).cpu().sum().item()
            loss_list.append(loss2.item())

    clean_acc = 100.0 * (float(correct_clean)/float(clean_size)) if clean_size else 0.0
    poison_acc = 100.0 * (float(correct_poison)/float(poison_size)) if poison_size else 0.0
    
    training_avg_loss = sum(loss_list)/len(loss_list)
    # training_avg_loss = 0.0
    if atkmodel_train:
        logger.info(colored("Training loss = {:.2f}, acc = {:.2f} of atk model this epoch".format(training_avg_loss, poison_acc), "yellow"))
    else:
        logger.info(colored("Training loss = {:.2f}, acc = {:.2f} of cls model this epoch".format(training_avg_loss, clean_acc), "yellow"))
        logger.info("Training clean_acc is {:.2f}, poison_acc = {:.2f}".format(clean_acc, poison_acc))
    del wg_clone    
     
def train_baseline(model, optimizer, train_loader, criterion, dataset, device, target_transform=None, local_e=0):
    loss_fn = nn.CrossEntropyLoss()
    func_fn = loss_fn
    total_loss = 0
    correct = 0
    
    correct_clean = 0
    correct_poison = 0
    total_loss = 0
    poison_data_count = 0
    
    dataset_size = 0
    poison_size = 0
    poison_data_count = 0
    clean_size = 0
    loss_list = []
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        bs = len(batch)
        data, targets, poison_num, _, _ = get_poison_batch(batch, dataset, device, target_transform=target_transform)
        # print(f"training targets: {targets}")
        optimizer.zero_grad()
        dataset_size += len(data)
        poison_data_count += poison_num

        output = model(data)
        class_loss = nn.functional.cross_entropy(output, targets)

        # distance_loss = helper.model_dist_norm_var(model, target_params_variables)
        # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
        # loss = helper.params['alpha_loss'] * class_loss + \
        #         (1 - helper.params['alpha_loss']) * distance_loss
        loss = class_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
    logger.info("Total loss is {:.2f}, training acc is {:.2f}".format(total_l, acc))

def test(model, device, test_loader, test_batch_size, criterion, mode="raw-task", dataset="cifar10", poison_type="fashion"):
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
    return final_acc, task_acc

def lira_test(args, device, atkmodel, scratchmodel, target_transform, 
         train_loader, test_loader, epoch, trainepoch, clip_image, 
         testoptimizer=None, log_prefix='Internal', epochs_per_test=5):
    #default phase 2 parameters to phase 1 
    
    if args['test_alpha'] is None:
        args['test_alpha'] = args['attack_alpha']
    if args['test_eps'] is None:
        args['test_eps'] = args['eps']
        
    print(f"\n------------------------------------\nStart postraining based on trajectories")
    
    atkmodel.eval()

    if testoptimizer is None:
        testoptimizer = optim.SGD(scratchmodel.parameters(), lr=args['lr'])
    for cepoch in range(trainepoch):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for batch_idx, (data, target) in pbar:
            bs = data.size(0)
            data, target = data.to(device), target.to(device)
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args['test_eps']
                atkdata = clip_image(data + noise)
                atktarget = target_transform(target)
                if args['attack_portion'] < 1.0:
                    atkdata = atkdata[:int(args['attack_portion']*bs)]
                    atktarget = atktarget[:int(args['attack_portion']*bs)]

            atkoutput = scratchmodel(atkdata)
            output = scratchmodel(data)
            
            loss_clean = loss_fn(output, target)
            loss_poison = loss_fn(atkoutput, atktarget)
            
            loss = args['attack_alpha'] * loss_clean + (1-args['test_alpha']) * loss_poison
            
            loss.backward()
            testoptimizer.step()
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
                pbar.set_description(
                    'Test [{}-{}] Loss: Clean {:.4f} Poison {:.4f} Total {:.5f}'.format(
                        epoch, cepoch+1,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item()
                    ))
        print(f"\n------------------------------------Start testing the model") 
        if cepoch % epochs_per_test == 0 or cepoch == trainepoch-1:
            correct = 0    
            correct_transform = 0
            test_loss = 0
            test_transform_loss = 0

            with torch.no_grad():
                for data, target in test_loader:
                    bs = data.size(0)
                    data, target = data.to(device), target.to(device)
                    output = scratchmodel(data)
                    test_loss += loss_fn(output, target).item() * bs  # sum up batch loss
                    pred = output.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    noise = atkmodel(data) * args['test_eps']
                    atkdata = clip_image(data + noise)
                    atkoutput = scratchmodel(atkdata)
                    test_transform_loss += loss_fn(atkoutput, target_transform(target)).item() * bs  # sum up batch loss
                    atkpred = atkoutput.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct_transform += atkpred.eq(
                        target_transform(target).view_as(atkpred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)

            print(
                '\n{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
                    log_prefix, cepoch, 
                    test_loss, test_transform_loss,
                    correct, correct_transform
                ))
        # print(f"\nFinish the post testing------------------------------------") 
    
 
    return correct, correct_transform

def test_updated(args, device, atkmodel, model, target_transform, 
         train_loader, test_loader, epoch, trainepoch, clip_image, 
         testoptimizer=None, log_prefix='Internal', epochs_per_test=3, 
         dataset="cifar10", criterion=None, subpath_saved="", test_eps=None, 
         is_poison=False, ident="global", atk_baseline=False):

    if args['test_alpha'] is None:
        args['test_alpha'] = args['attack_alpha']
    if args['test_eps'] is None:
        args['test_eps'] = args['eps']
    if not test_eps:
        test_eps = args['test_eps']
    # import IPython
    # IPython.embed()
    # start_time = time.time()
    
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
    
    if is_poison:
        csv_record.test_result.append([ident, epoch, test_loss, test_transform_loss, correct, correct_transform])
        
    print(f"Time for end testing: {time.time() - start_time}")
    return correct, correct_transform
    
class FederatedLearningTrainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class FrequencyFederatedLearningTrainer(FederatedLearningTrainer):
    def __init__(self, arguments=None, lira_args=None, *args, **kwargs):
        #self.poisoned_emnist_dataset = arguments['poisoned_emnist_dataset']
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        # self.federated_optimizer = arguments['federated_optimizer'] # default: FedAvg
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacking_fl_rounds = arguments['attacking_fl_rounds']
        self.poisoned_emnist_train_loader = arguments['poisoned_emnist_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.save_model = arguments['save_model']
        # self.num_dps_poisoned_dataset = int(arguments['num_dps_poisoned_dataset'] * (1.0+lira_args['attack_portion']))
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.scale_weights_poison = arguments['scale_weights_poison']
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.attack_method = arguments["attack_method"]
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.eps_decay = arguments['eps_decay']
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.atk_lr = arguments['atk_lr']
        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']
        self.atkmodel = arguments['atkmodel']
        self.tgtmodel = arguments['tgtmodel']
        self.lira_args = lira_args
        self.baseline = arguments['baseline']
        self.retrain = arguments['retrain']
        self.aggregator = arguments['aggregator'] # default: fedavg
        # self.create_net = arguments['create_net']
        self.scratch_model = arguments['scratch_model']
        self.atk_eps = arguments['atk_eps']
        self.atk_test_eps = arguments['atk_test_eps']
        self.scale_factor = arguments['scale']
        self.mask_ratio = 0.30 #TODO: modify it later, used only for Neurotoxin attack strategy
        self.flatten_net_avg = None
        self.folder_path = arguments['folder_path']
        self.historical_grad_mask = None
        self.atk_baseline = arguments['atk_baseline']

        if self.attack_method == "pgd":
            self.pgd_attack = True
            self.neurotoxin_attack = False
        elif self.attack_method == "neurotoxin":
            self.neurotoxin_attack = True
            self.pgd_attack = False
        else:
            self.pgd_attack = False
            self.neurotoxin_attack = False

        if arguments["defense_technique"] == "no-defense":
            self._defender = None
        elif arguments["defense_technique"] == "norm-clipping" or arguments["defense_technique"] == "norm-clipping-adaptive":
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "weak-dp":
            # doesn't really add noise. just clips
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "krum":
            self._defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "multi-krum":
            self._defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "rfa":
            self._defender = RFA()
        elif arguments["defense_technique"] == "crfl":
            self._defender = CRFL()
        elif arguments["defense_technique"] == "rlr":
            pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            args_rlr={
                'aggr':'avg',
                'noise':0,
                'clip': 0,
                'server_lr': self.args_lr,
            }
            # theta = 2
            theta = 1
            self._defender = RLR(n_params=pytorch_total_params, device=self.device, args=args_rlr, robustLR_threshold=theta)
        elif arguments["defense_technique"] == "foolsgold":
            pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            self._defender = FoolsGold(num_clients=self.part_nets_per_round, num_classes=10, num_features=pytorch_total_params)
        else:
            NotImplementedError("Unsupported defense method !")

    def run(self, wandb_ins):
        expected_atk_threshold = 0.85
        cnt_masks = 0
        print(f"self.scale_weights_poison: {self.scale_weights_poison}")
        # if self.dataset == "mnist":
        #     expected_atk_threshold = 0.90
        # elif self.dataset == "cifar10":
        #     expected_atk_threshold = 0.90
            
        main_task_acc = []
        raw_task_acc = []
        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        print(colored('Under defense technique = {}'.format(self.defense_technique), 'red'))
        
        # variables for LIRA algorithms only
        trainlosses = []
        best_acc_clean = 0
        best_acc_poison = 0
        clip_image = get_clip_image(self.dataset)
        attack_train_epoch = self.lira_args['train_epoch']
        target_transform = get_target_transform(self.lira_args)
        
        pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
        # The number of previous iterations to use FoolsGold on
        memory_size = 0
        delta_memory = np.zeros((self.num_nets, pytorch_total_params, memory_size))
        summed_deltas = np.zeros((self.num_nets, pytorch_total_params))
        
        basepath, checkpoint_path, bestmodel_path = create_paths(self.lira_args)
        print('========== PATHS ==========')
        print(f'Basepath: {basepath}')
        print(f'Checkpoint Model: {checkpoint_path}')
        print(f'Best Model: {bestmodel_path}')
        
        # LOAD_ATK_MODEL = False
        LOAD_ATK_MODEL = True
        checkpoint_path = bestmodel_path
        if os.path.exists(checkpoint_path) and not self.retrain:
            # Load previously saved models
            checkpoint = torch.load(checkpoint_path)
            print(colored('Load existing attack model from path {}'.format(checkpoint_path), 'red'))
            self.atkmodel.load_state_dict(checkpoint['atkmodel'], strict=True)
            atk_model_loaded = True
        else:
            # Create new model
            print(colored('Create new model from {}'.format(checkpoint_path), 'blue'))
            best_acc_clean = 0
            best_acc_poison = 0
            
        subpath_trigger_saved = f"{self.dataset}baseline_{self.baseline}_atkepc_{attack_train_epoch}_eps_{self.atk_eps}_test_eps_{self.atk_test_eps}"
        if self.baseline:
            subpath_trigger_saved = ""
        
        start_time = time.time()
        
        acc_clean, backdoor_acc = test_updated(self.lira_args, self.device, self.atkmodel, self.net_avg, target_transform, 
                                self.clean_train_loader, self.vanilla_emnist_test_loader, 0, self.lira_args['train_epoch'], clip_image, 
                                testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, 
                                subpath_saved=subpath_trigger_saved, is_poison=True, atk_baseline=self.atk_baseline)
        
        print(f"\n----------TEST FOR GLOBAL MODEL BEFORE FEDERATED TRAINING----------------")
        print(f"Main task acc: {round(acc_clean*100, 2)}%")
        print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
        print(f"--------------------------------------------------------------------------\n")
        
        
        # Try to save models at different fl rounds
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        folder_path = f'saved_models/{self.folder_path}_{"baseline" if self.baseline else "poison"}_{current_time}'
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.info(f'current path for saving: {folder_path}')


        local_acc_clean, local_acc_poison = 0.0, 0.0
        best_main_acc = 0.0
        # let's conduct multi-round training
        cur_training_eps = self.atk_eps
        start_decay_r = 0
        start_training_atk_model = True
        tgt_optimizer = optim.Adam(self.tgtmodel.parameters(), lr=self.atk_lr)
        atk_optimizer = optim.Adam(self.atkmodel.parameters(), lr=self.atk_lr)
        self.alternative_training = False
        self.flatten_net_avg = flatten_model(copy.deepcopy(self.net_avg))
        
        # end_time = time.time()
        elapsed_time = time.time() - start_time
        # print time by # seconds
        print(f"Time for TEST FOR GLOBAL MODEL BEFORE FEDERATED TRAINING: {elapsed_time}")
        # print(f"Time for TEST FOR GLOBAL MODEL BEFORE FEDERATED TRAINING: {(end_time - start_time).seconds}")
        
        print("----------START FEDERATED TRAINING----------------")
        
        for flr in range(1, self.fl_round+1):
            print("---"*30)
            print("Start communication round {}".format(flr))
            
            if local_acc_poison >= expected_atk_threshold and self.retrain:
                self.retrain = False
                print(colored('Starting sub-training phase from flr = {}'.format(flr), 'red'))
                self.lira_args["test_eps"] = self.atk_test_eps
                start_decay_r = flr
            
            if self.retrain == False:
                cur_training_eps = max(self.atk_test_eps, exponential_decay(self.atk_eps, self.eps_decay, flr-start_decay_r))
                # self.alternative_training = !self.alternative_training
            g_user_indices = []

            if self.defense_technique == "norm-clipping-adaptive":
                # experimental
                norm_diff_collector = []
            
            # tau for FedNova
	        # rho: Parameter controlling the momentum SGD
            list_ni = [] # number of data points for each client
            list_tau = [] # tau for FedNova (number of batch training = batch * epoch)
            list_ai = [] # weight for FedNova

            # if self.neurotoxin_attack and ((backdoor_acc > 0.95 and cur_training_eps <= self.atk_test_eps)):
            if self.neurotoxin_attack and flr >= 400:
                self.baseline = True
                logger.info(colored(f"Stop poisoning from round {flr}!!!", "red"))    
            # if self.atk_baseline and flr >= 400:
            #     self.baseline = True
            #     logger.info(colored(f"Stop poisoning from round {flr}!!!", "red"))  
             
            if flr in self.attacking_fl_rounds and not self.baseline:
                # randomly select participating clients
                selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
                num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
                total_num_dps_per_round = sum(num_data_points)
                
                if not self.retrain:
                    self.alternative_training = not self.alternative_training

                # net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
                # logger.info("Net freq: {}, FL round: {} without adversary".format(net_freq, flr))

                # we need to reconstruct the net list at the beginning
                net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                logger.info("################## Starting fl round: {}".format(flr))

                # assume the first client is the attacker
                num_data_points[0] = self.num_dps_poisoned_dataset
                net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
                logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr))
                # FEDNOVA
                for id_client in range(self.part_nets_per_round):
                    if id_client == 0:
                        list_tau.append(np.ceil(num_data_points[id_client] / self.batch_size) * self.adversarial_local_training_period )
                    else:
                        list_tau.append(np.ceil(num_data_points[id_client] / self.batch_size) * self.local_training_period )
                    list_ni.append(num_data_points[id_client])
                #pdb.set_trace()

                # we need to reconstruct the net list at the beginning
                # net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                # logger.info("################## Starting fl round: {}".format(flr))
                
                model_original = list(self.net_avg.parameters())
                # init_local_model = copy.deepcopy(self.net_avg)
                # super hacky but I'm doing this for the prox-attack
                wg_clone = copy.deepcopy(self.net_avg)
                # wg_hat = None
                # v0 = torch.nn.utils.parameters_to_vector(model_original)
                # wg_norm_list.append(torch.norm(v0).item())
                
                """ Start the FL process """
                for net_idx, net in enumerate(net_list):
                    init_local_model = dict()
                    for name, data in wg_clone.state_dict().items():
                        init_local_model[name] = wg_clone.state_dict()[name].clone()
                    #net  = net_list[net_idx]                
                    # if net_idx == 0:
                    #     global_user_idx = -1 # we assign "-1" as the indices of the attacker in global user indices
                    #     pass
                    # else:
                    #     global_user_idx = selected_node_indices[net_idx-1]
                    #     dataidxs = self.net_dataidx_map[global_user_idx]
                    #     train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                    #                                     self.test_batch_size, dataidxs) # also get the data loader
                    global_user_idx = selected_node_indices[net_idx]
                    dataidxs = self.net_dataidx_map[global_user_idx] # --> keep the same data for an attacker 

                    # if self.attack_case == "edge-case":
                    train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                    self.test_batch_size, dataidxs) # also get the data loader
                    # else:
                    #     NotImplementedError("Unsupported attack case ...")
                    if net_idx == 0:
                        train_dl_local = copy.deepcopy(self.clean_train_loader)
                    g_user_indices.append(global_user_idx)
                    
                    if net_idx == 0:
                        logger.info("@@@@@@@@ Working on client: {}, which is Attacker".format(net_idx))
                    else:
                        logger.info("@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                    adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # looks like adversary needs same lr to hide with others
                    # prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)

                    for param_group in optimizer.param_groups:
                        logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))
                    
                    if net_idx == 0:
                        """ for attacker only """
                        attack_portion = 0.0 if self.retrain else self.lira_args['attack_portion']
                        atk_alpha = self.lira_args['attack_alpha'] if not self.retrain else 1.0
                        
                        print(colored(f"At flr {flr}, test eps is {cur_training_eps}", "green"))

                        mask_grad_list = []
                        # if self.neurotoxin_attack and cur_training_eps != self.atk_eps:
                        pgd_attack = False if cur_training_eps == self.atk_eps else self.pgd_attack
                        clone_local_model = copy.deepcopy(net)

                        
                        for e in range(1, self.adversarial_local_training_period+1):
                            if not self.atk_baseline:
                                pdg_eps = self.eps*self.args_gamma**(flr-start_decay_r) if self.defense_technique in ('krum', 'multi-krum') else self.eps
                                train_lira(net, self.atkmodel, self.tgtmodel, optimizer, atk_optimizer, train_dl_local,
                                        self.criterion, cur_training_eps, atk_alpha, attack_portion, clip_image, 
                                        target_transform, 1, atkmodel_train=False, device=self.device,
                                        pgd_attack=pgd_attack, batch_idx=e, project_frequency=self.project_frequency,
                                        pgd_eps=pdg_eps, model_original=wg_clone, adv_optimizer=adv_optimizer, 
                                        mask_grad_list=mask_grad_list, aggregator=self.aggregator, wg_hat=wg_clone, local_e=e)
                            else:
                                train_baseline(net, optimizer, train_dl_local, self.criterion, self.dataset, self.device, target_transform, e)
                        
                        # mask_flat_all_layer = get_grad_mask_F3BA(clone_local_model, wg_clone, None, None)

                        # Retrain the attack model to avoid catastrophic forgetting            
                        if (self.retrain or self.alternative_training) and not self.atk_baseline:
                            # cp_net = copy.deepcopy(net)
                            # lira_train_eps = self.atk_eps if self.alternative_training else cur_training_eps
                            lira_train_eps = cur_training_eps
                            for e in range(1, attack_train_epoch+1):
                                train_lira(net, self.atkmodel, self.tgtmodel, None, atk_optimizer, train_dl_local,
                                        criterion, lira_train_eps, atk_alpha, 1.0, clip_image, target_transform, 1, 
                                        atkmodel_train=True, device=self.device, pgd_attack=False)
                                self.tgtmodel.load_state_dict(self.atkmodel.state_dict())

                        replacement_flr = 400 if self.dataset == "mnist" else 500
                        if self.model_replacement and cur_training_eps != self.atk_eps and flr >= replacement_flr:
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))
                            # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                            # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                            for idx, param in enumerate(net.parameters()):
                                param.data = (param.data - model_original[idx])*(total_num_dps_per_round/self.num_dps_poisoned_dataset) + model_original[idx]
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                        local_acc_clean, local_acc_poison = test_updated(self.lira_args, self.device, self.tgtmodel, net, target_transform, 
                            self.clean_train_loader, self.vanilla_emnist_test_loader, e, self.lira_args['train_epoch'], clip_image,
                            log_prefix='Internal', dataset=self.dataset, criterion=criterion, subpath_saved=subpath_trigger_saved, test_eps=cur_training_eps)
                        
                        if self.scale_weights_poison != 1.0 and flr%501 == 0:
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker before scaling with factor {}: Norm = {}".format(self.scale_weights_poison, torch.norm(v)))
                            # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                            # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                            for idx, param in enumerate(net.parameters()):
                                param.data = (param.data - model_original[idx])*self.scale_weights_poison + model_original[idx]
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                        local_acc_clean, local_acc_poison = test_updated(self.lira_args, self.device, self.tgtmodel, net, target_transform, 
                            self.clean_train_loader, self.vanilla_emnist_test_loader, flr, self.lira_args['train_epoch'], clip_image,
                            log_prefix='Internal', dataset=self.dataset, criterion=criterion, subpath_saved=subpath_trigger_saved, 
                            test_eps=cur_training_eps, is_poison=True, ident=global_user_idx, atk_baseline = self.atk_baseline)
                        
                    else:
                        """ for normal client only """
                        for e in range(1, self.local_training_period+1):
                            train_lira(net, self.atkmodel, self.tgtmodel, optimizer, None, train_dl_local,
                                        self.criterion, cur_training_eps, 1.0, 0.0, clip_image, target_transform, 1, 
                                        atkmodel_train=False, device=self.device, aggregator=self.aggregator, wg_hat=wg_clone)
                    
            else:
                wg_clone = copy.deepcopy(self.net_avg)
                      
                selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
                num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
                total_num_dps_per_round = sum(num_data_points)
                
                # FEDNOVA
                for id_client in range(self.part_nets_per_round):
                    list_tau.append(np.ceil(num_data_points[id_client] / self.batch_size) * self.local_training_period )
                    list_ni.append(num_data_points[id_client])
                  
                net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
                logger.info("Net freq: {}, FL round: {} without adversary".format(net_freq, flr))

                # we need to reconstruct the net list at the beginning
                net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                logger.info("################## Starting fl round: {}".format(flr))

                # start the FL process
                
                # print(f"No attacker in this round {flr} with list client: {net_list}")
                
                for net_idx, net in enumerate(net_list):
                    start_time = time.time()
                    
                    global_user_idx = selected_node_indices[net_idx]
                    dataidxs = self.net_dataidx_map[global_user_idx] # --> keep the same data for an attacker 

                    # if self.attack_case == "edge-case":
                    train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                    self.test_batch_size, dataidxs) # also get the data loader
                    # else:
                    #     NotImplementedError("Unsupported attack case ...")
                    # if net_idx == 0:
                    #     train_dl_local = copy.deepcopy(self.clean_train_loader)
                    g_user_indices.append(global_user_idx)
                    
                    logger.info("@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                    for param_group in optimizer.param_groups:
                        logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))                    
                  
                    """ for normal client only """
                    for e in range(1, self.local_training_period+1):
                        train_lira(net, self.atkmodel, self.tgtmodel, optimizer, atk_optimizer, train_dl_local,
                                    self.criterion, cur_training_eps, 1.0, 0.0, clip_image, target_transform, 1, 
                                    atkmodel_train=False, device=self.device, aggregator=self.aggregator, wg_hat=wg_clone)
                    
                    honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(honest_norm_diff)   
                    end_time = time.time()
                    logger.info("Time for local training client {}: {}".format(net_idx, end_time-start_time))

                adv_norm_diff_list.append(0)
                model_original = list(self.net_avg.parameters())
                v0 = torch.nn.utils.parameters_to_vector(model_original)
                wg_norm_list.append(torch.norm(v0).item())
            
            
            ### conduct defense here:
            print(colored(f"Conduct defense technique: {self.defense_technique}", "red"))
            if self.defense_technique == "foolsgold":
                delta = np.zeros((self.num_nets, pytorch_total_params))
                if memory_size > 0:
                    for net_idx, global_client_indx in enumerate(selected_node_indices):
                        flatten_local_model = flatten_model(net_list[net_idx])
                        local_update = flatten_local_model - self.flatten_net_avg
                        delta[global_client_indx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[global_client_indx, :]) > 1:
                            delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(delta[global_client_indx, :])

                        delta_memory[global_client_indx, :, flr % memory_size] = delta[global_client_indx, :]
                    # Track the total vector from each individual client
                    summed_deltas = np.sum(delta_memory, axis=2)      
                else:
                    for net_idx, global_client_indx in enumerate(selected_node_indices):
                        flatten_local_model = flatten_model(net_list[net_idx])
                        local_update = flatten_local_model - self.flatten_net_avg
                        local_update = local_update.detach().cpu().numpy()
                        delta[global_client_indx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[global_client_indx, :]) > 1:
                            delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(delta[global_client_indx, :])
                    # Track the total vector from each individual client
                    # print(f"delta={delta[selected_node_indices,:]}")
                    # print(f"summed_deltas[selected_node_indices,:].shape is: {summed_deltas[selected_node_indices,:].shape}")

                    summed_deltas[selected_node_indices,:] = summed_deltas[selected_node_indices,:] + delta[selected_node_indices,:]
                    # print(f"summed_deltas.shape is: {summed_deltas.shape}")
                    # print(f"summed_deltas={summed_deltas[selected_node_indices,:]}")
            if self.defense_technique == "no-defense":
                pass
            elif self.defense_technique == "norm-clipping":
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "norm-clipping-adaptive":
                # we will need to adapt the norm diff first before the norm diff clipping
                logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector, 
                    np.mean(norm_diff_collector)))
                self._defender.norm_bound = np.mean(norm_diff_collector)
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "weak-dp":
                # this guy is just going to clip norm. No noise added here
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net,
                                        global_model=self.net_avg,)
            elif self.defense_technique == "krum":
                print("start performing krum...")
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=self.device)
                print(f"net_list = {len(net_list)}, net_freq={net_freq}")
            elif self.defense_technique == "multi-krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=self.device)
            elif self.defense_technique == 'rlr':
                print(f"num_data_points: {num_data_points}")
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        num_dps=num_data_points,
                                                        global_model=copy.deepcopy(self.net_avg))
            elif self.defense_technique == "foolsgold":
                print(f"Performing foolsgold...")
                # net_list, net_freq = self._defender.exec(client_models=net_list, delta = delta[selected_node_indices,:] ,summed_deltas=summed_deltas[selected_node_indices,:], net_avg=self.net_avg, r=flr, device=self.device)
                wv = self._defender.exec(client_models=net_list, delta = delta[selected_node_indices,:] ,summed_deltas=summed_deltas[selected_node_indices,:], net_avg=self.net_avg, r=flr, device=self.device)
                net_freq = wv
                print(f"net_freq: {net_freq}")
            elif self.defense_technique == "rfa":
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        net_freq=net_freq,
                                                        maxiter=500,
                                                        eps=1e-5,
                                                        ftol=1e-7,
                                                        device=self.device)
            elif self.defense_technique == "crfl":
                cp_net_avg = copy.deepcopy(self.net_avg)
                # pseudo_global_net = fed_avg_aggregator(cp_net_avg, net_list, net_freq, device=self.device, model=self.model)
                net_list, net_freq = self._defender.exec(target_model=cp_net_avg,
                                                         epoch=flr,
                                                         sigma_param=0.01,
                                                         dataset_name=self.dataset,
                                                         device=self.device)
            else:
                NotImplementedError("Unsupported defense method !")
            
            
            # after local training periods
            fed_aggregator = self.aggregator
            print(f"\nStart performing fed aggregator: {fed_aggregator}...")
            fed_nova_rho = 0.1
            list_ai = [(tau - fed_nova_rho * (1 - pow(fed_nova_rho, tau)) / (1 - fed_nova_rho)) / (1 - fed_nova_rho) for tau in list_tau]
            
            if fed_aggregator == "fedavg":
                self.net_avg = fed_avg_aggregator(self.net_avg, net_list, net_freq, device=self.device, model=self.model)
            elif fed_aggregator == "fednova":
                self.net_avg = fed_nova_aggregator(self.net_avg, net_list, list_ni, self.device, list_ai)
            else:
                NotImplementedError(f"Unsupported fed agg {fed_aggregator} method !")

            
            acc_clean, backdoor_acc = test_updated(args=self.lira_args, device=self.device, atkmodel=self.tgtmodel, model=self.net_avg, 
                                                   target_transform=target_transform, train_loader=self.clean_train_loader, 
                                                   test_loader=self.vanilla_emnist_test_loader,epoch=flr, trainepoch=self.lira_args['train_epoch'], 
                                                   clip_image=clip_image, testoptimizer=None, log_prefix='External', epochs_per_test=3, 
                                                   dataset=self.dataset, criterion=criterion, subpath_saved=subpath_trigger_saved, test_eps=cur_training_eps,
                                                   is_poison=True, atk_baseline=self.atk_baseline)
            
            
            if self.defense_technique == "weak-dp":
                # add noise to self.net_avg
                noise_adder = AddNoise(stddev=self.stddev)
                noise_adder.exec(client_model=self.net_avg,
                                                device=self.device)

            v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
            logger.info("############ Averaged Model : Norm {}".format(torch.norm(v)))

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.vanilla_model, epoch=0, fl_round=flr, mode="avg")
            
            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            # START SAVING THE MODEL FOR FURTHER ANALYSIS
            model_name = '{0}/model_last_flr_{1}.pt'.format(folder_path, flr)
            atk_model_name = '{0}/atkmodel_last_flr_{1}.pt'.format(folder_path, flr)
            if flr >= 200 and flr%50 == 0 and self.save_model:
                torch.save(self.net_avg.state_dict(), model_name)
                torch.save(self.tgtmodel.state_dict(), atk_model_name)
            
            wandb_logging_items = {
                'fl_iter': flr,
                'main_task_acc': acc_clean*100.0,
                'backdoor_task_acc': backdoor_acc*100.0, 
                'local_best_acc_clean': best_acc_clean,
                'local_best_acc_poison': best_acc_poison,
                'local_MA': local_acc_clean*100.0,
                'local_BA': local_acc_poison*100.0,
                'cur_training_eps': cur_training_eps,
            }
            if wandb_ins:
                print(f"start logging to wandb")
                wandb_ins.log({"General Information": wandb_logging_items})
            csv_record.save_result_csv(flr, True, folder_path)
            raw_acc = 0
            overall_acc = acc_clean
            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            raw_task_acc.append(raw_acc)
            backdoor_task_acc.append(backdoor_acc)


        saved_path = "checkpoint/last_model"
        torch.save(self.net_avg.state_dict(), f"{saved_path}/lira_{self.dataset}_{self.model}_{self.atk_test_eps}.pt")

        df = pd.DataFrame({'fl_iter': fl_iter_list, 
                            'main_task_acc': main_task_acc, 
                            'backdoor_acc': backdoor_task_acc, 
                            'raw_task_acc':raw_task_acc, 
                            'adv_norm_diff': adv_norm_diff_list, 
                            'wg_norm': wg_norm_list
                            })
       
        if self.poison_type == 'ardis':
            # add a row showing initial accuracies
            df1 = pd.DataFrame({'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]})
            df = pd.concat([df1, df])

        results_filename = get_results_filename(self.poison_type, self.attack_method, self.model_replacement, self.project_frequency,
                self.defense_technique, self.norm_bound, self.prox_attack, False, self.model)

        df.to_csv(results_filename, index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))


class FixedPoolFederatedLearningTrainer(FederatedLearningTrainer):
    def __init__(self, arguments=None, lira_args=None, *args, **kwargs):
        #self.poisoned_emnist_dataset = arguments['poisoned_emnist_dataset']
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacker_pool_size = arguments['attacker_pool_size']
        self.poisoned_emnist_train_loader = arguments['poisoned_emnist_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.attack_method = arguments["attack_method"]
        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.eps_decay = arguments['eps_decay']
        self.scale_weights_poison = arguments['scale_weights_poison']
        self.dataset = arguments["dataset"]
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.atk_lr = arguments['atk_lr']

        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']
        self.atkmodel = arguments['atkmodel']
        self.tgtmodel = arguments['tgtmodel']
        self.lira_args = lira_args
        self.baseline = arguments['baseline']
        self.retrain = arguments['retrain']
        self.scratch_model = arguments['scratch_model']
        self.atk_eps = arguments['atk_eps']
        self.atk_test_eps = arguments['atk_test_eps']
        self.scale_factor = arguments['scale']
        self.flatten_net_avg = None

        logger.info("Posion type! {}".format(self.poison_type))

        if self.attack_method == "pgd":
            self.pgd_attack = True
        elif self.attack_method == "neurotoxin":
            self.neurotoxin_attack = True
        else:
            self.pgd_attack = False
            self.neurotoxin_attack = False


        if arguments["defense_technique"] == "no-defense":
            self._defender = None
        elif arguments["defense_technique"] == "norm-clipping" or arguments["defense_technique"] == "norm-clipping-adaptive":
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "weak-dp":
            # doesn't really add noise. just clips
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "krum":
            self._defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "multi-krum":
            self._defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "rfa":
            self._defender = RFA()
        elif arguments["defense_technique"] == "rlr":
            pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            args_rlr={
                'aggr':'avg',
                'noise':0,
                'clip': 0,
                'server_lr': self.args_lr,
            }
            # theta = 4 if arguments['dataset'] == 'cifar10' else 8
            # theta = int(arguments['part_nets_per_round']*self.attacker_pool_size/self.num_nets)+1
            theta = 8
            self._defender = RLR(n_params=pytorch_total_params, device=self.device, args=args_rlr, robustLR_threshold=theta)
        elif arguments["defense_technique"] == "foolsgold":
            pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            self._defender = FoolsGold(num_clients=self.part_nets_per_round, num_classes=10, num_features=pytorch_total_params)
        else:
            NotImplementedError("Unsupported defense method !")

        self.__attacker_pool = np.random.choice(self.num_nets, self.attacker_pool_size, replace=False)

    def run(self, wandb_ins):
        expected_atk_threshold = 0.90
        # if self.dataset == "mnist":
        #     expected_atk_threshold = 0.90
        # elif self.dataset == "cifar10":
        #     expected_atk_threshold = 0.90
        main_task_acc = []
        raw_task_acc = []
        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        print(colored('Under defense technique = {}'.format(self.defense_technique), 'red'))
        
        # variables for LIRA algorithms only
        trainlosses = []
        best_acc_clean = 0
        best_acc_poison = 0
        avoid_cls_reinit = True
        clip_image = get_clip_image(self.dataset)
        attack_train_epoch = self.lira_args['train_epoch']
        target_transform = get_target_transform(self.lira_args)
        tgt_optimizer = None
        loss_list = []

        pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
        # The number of previous iterations to use FoolsGold on
        memory_size = 0
        delta_memory = np.zeros((self.num_nets, pytorch_total_params, memory_size))
        summed_deltas = np.zeros((self.num_nets, pytorch_total_params))
        
        basepath, checkpoint_path, bestmodel_path = create_paths(self.lira_args)
        print('========== PATHS ==========')
        print(f'Basepath: {basepath}')
        print(f'Checkpoint Model: {checkpoint_path}')
        print(f'Best Model: {bestmodel_path}')
        
        # LOAD_ATK_MODEL = False
        LOAD_ATK_MODEL = True
        checkpoint_path = bestmodel_path
        if os.path.exists(checkpoint_path) and not self.retrain:
            # Load previously saved models
            checkpoint = torch.load(checkpoint_path)
            print(colored('Load existing attack model from path {}'.format(checkpoint_path), 'red'))
            self.atkmodel.load_state_dict(checkpoint['atkmodel'], strict=True)
            atk_model_loaded = True
        else:
            # Create new model
            print(colored('Create new model from {}'.format(checkpoint_path), 'blue'))
            best_acc_clean = 0
            best_acc_poison = 0
            trainlosses = []
            start_epoch = 1
            
        subpath_trigger_saved = f"{self.dataset}baseline_{self.baseline}_atkepc_{attack_train_epoch}_eps_{self.atk_eps}_test_eps_{self.atk_test_eps}"
        if self.baseline:
            subpath_trigger_saved = ""
            
        acc_clean, backdoor_acc = test_updated(self.lira_args, self.device, self.atkmodel, self.net_avg, target_transform, 
                                self.clean_train_loader, self.vanilla_emnist_test_loader, 0, self.lira_args['train_epoch'], clip_image, 
                                testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)
        
        print(f"\n----------TEST FOR GLOBAL MODEL BEFORE FEDERATED TRAINING----------------")
        print(f"Main task acc: {round(acc_clean*100, 2)}%")
        print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
        print(f"--------------------------------------------------------------------------\n")
        local_acc_clean, local_acc_poison = 0.0, 0.0
        best_main_acc = 0.0
        # let's conduct multi-round training
        cur_training_eps = self.atk_eps
        start_decay_r = 0
        start_training_atk_model = True
        tgt_optimizer = optim.Adam(self.tgtmodel.parameters(), lr=self.atk_lr)
        atk_optimizer = optim.Adam(self.atkmodel.parameters(), lr=self.atk_lr)
        self.alternative_training = False
        self.flatten_net_avg = flatten_model(self.net_avg)
        list_ni = []
        list_tau = []
        list_ai = []
        
        for flr in range(1, self.fl_round+1):
            # randomly select participating clients
            # in this current version, we sample `part_nets_per_round` per FL round since we assume attacker will always participates
            if local_acc_poison >= expected_atk_threshold and self.retrain:
                self.retrain = False
                print(colored('Starting sub-training phase from flr = {}'.format(flr), 'red'))
                self.lira_args["test_eps"] = self.atk_test_eps
                start_decay_r = flr
            
            if self.retrain == False:
                cur_training_eps = max(self.atk_test_eps, exponential_decay(self.atk_eps, self.eps_decay, flr-start_decay_r))
            g_user_indices = []

            selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
            selected_attackers = [idx for idx in selected_node_indices if idx in self.__attacker_pool]
            selected_honest_users = [idx for idx in selected_node_indices if idx not in self.__attacker_pool]
            logger.info("Selected Attackers in FL iteration-{}: {}".format(flr, selected_attackers))

            num_data_points = []
            for sni in selected_node_indices:
                if sni in selected_attackers:
                    num_data_points.append(self.num_dps_poisoned_dataset)
                else:
                    num_data_points.append(len(self.net_dataidx_map[sni]))

            total_num_dps_per_round = sum(num_data_points)
            net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
            logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr)) 

            # we need to reconstruct the net list at the beginning
            total_attackers = len(selected_attackers)
            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
            # atk_model_list = [copy.deepcopy(self.atkmodel) for _ in range(total_attackers)] # copy list of attack model to train independently
            atk_model_list = {g_id: copy.deepcopy(self.atkmodel) for g_id in selected_attackers} # copy list of attack model to train independently

            logger.info("################## Starting fl round: {}".format(flr))
            model_original = list(self.net_avg.parameters())
            # super hacky but I'm doing this for the prox-attack
            wg_clone = copy.deepcopy(self.net_avg)
            wg_hat = None
            v0 = torch.nn.utils.parameters_to_vector(model_original)
            wg_norm_list.append(torch.norm(v0).item())

            # calculate the tau for each client
            for sni in selected_node_indices:
                if sni in selected_attackers:
                    list_ni.append(self.num_dps_poisoned_dataset)
                    list_tau.append(np.ceil(self.num_dps_poisoned_dataset/ self.batch_size) * self.adversarial_local_training_period)
                else:
                    list_ni.append(len(self.net_dataidx_map[sni]))
                    list_tau.append(np.ceil(len(self.net_dataidx_map[sni])/ self.batch_size) * self.local_training_period)
                    

            #     # start the FL process
            for net_idx, global_user_idx in enumerate(selected_node_indices):
                net  = net_list[net_idx]
                if global_user_idx in selected_attackers:
                    pass
                else:
                    dataidxs = self.net_dataidx_map[global_user_idx]
         
                    # add p-percent edge-case attack here:
                    if self.attack_case == "edge-case":
                        train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                       self.test_batch_size, dataidxs) # also get the data loader
                    elif self.attack_case in ("normal-case", "almost-edge-case"):
                        train_dl_local, _ = get_dataloader_normal_case(self.dataset, './data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                    else:
                        NotImplementedError("Unsupported attack case ...")
                
                logger.info("@@@@@@@@ Working on client (global-index): {}, which {}-th user in the current round".format(global_user_idx, net_idx))

           
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # looks like adversary needs same lr to hide with others
                prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
                for param_group in optimizer.param_groups:
                    logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))


                current_adv_norm_diff_list = []
                if global_user_idx in selected_attackers:
                    """ for attacker only """
                    # local_atkmodel = atk_model_list[global_user_idx]
                    atk_optimizer = optim.Adam(atk_model_list[global_user_idx].parameters(), lr=self.atk_lr)
                    attack_portion = 0.0 if self.retrain else self.lira_args['attack_portion']
                    atk_alpha = self.lira_args['attack_alpha'] if not self.retrain else 1.0
                    
                    print(colored(f"At flr {flr}, test eps is {cur_training_eps}", "green"))

                 
                    if self.prox_attack:
                        # estimate w_hat
                        for inner_epoch in range(1, self.local_training_period+1):
                            estimate_wg(wg_clone, self.device, self.clean_train_loader, prox_optimizer, inner_epoch, log_interval=self.log_interval, criterion=self.criterion)
                        wg_hat = wg_clone

                 
                    """LIRA CODE"""
                    for e in range(1, self.adversarial_local_training_period+1):
                        #if self.defense_technique in ('krum', 'multi-krum'):  
                        pdg_eps = self.eps*self.args_gamma**(flr-1) if self.defense_technique in ('krum', 'multi-krum') else self.eps
                        
                        train_lira(net, atk_model_list[global_user_idx], self.tgtmodel, optimizer, 
                                atk_optimizer, self.clean_train_loader, self.criterion, cur_training_eps, 
                                atk_alpha, attack_portion, clip_image, target_transform, 1, 
                                atkmodel_train=False, device=self.device,
                                pgd_attack=self.pgd_attack, batch_idx=e, project_frequency=self.project_frequency,
                                pgd_eps=pdg_eps, model_original=model_original, adv_optimizer=adv_optimizer)
                    if self.retrain or self.alternative_training:
                        cp_net = copy.deepcopy(net)
                        for e in range(1, attack_train_epoch+1):
                            train_lira(net, atk_model_list[global_user_idx], self.tgtmodel, None, atk_optimizer, self.clean_train_loader,
                                    criterion, cur_training_eps, atk_alpha, 1.0, clip_image, target_transform, 1, 
                                    atkmodel_train=True, 
                                    device=self.device)

                    test(net, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
                    
                    # Testing for local model and local attack model
                    local_acc_clean, local_acc_poison = test_updated(self.lira_args, self.device, atk_model_list[global_user_idx], net, target_transform, 
                        self.clean_train_loader, self.vanilla_emnist_test_loader, e, self.lira_args['train_epoch'], clip_image,
                        log_prefix='Internal', dataset=self.dataset, criterion=criterion, subpath_saved=subpath_trigger_saved, test_eps=cur_training_eps)
                    # test(net, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)

                    # if model_replacement scale models
                    if self.model_replacement:
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))
                        # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                        # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                        for idx, param in enumerate(net.parameters()):
                            param.data = (param.data - model_original[idx])*(total_num_dps_per_round/self.num_dps_poisoned_dataset) + model_original[idx]
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                    # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                    current_adv_norm_diff_list.append(adv_norm_diff)

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(adv_norm_diff)
                else:
                    # for e in range(1, self.local_training_period+1):
                    #    train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)                
                    # # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                    # calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")

                    for e in range(1, self.local_training_period+1):
                        train_lira(net, self.atkmodel, self.tgtmodel, optimizer, atk_optimizer, train_dl_local,
                                    self.criterion, cur_training_eps, 1.0, 0.0, clip_image, target_transform, 1, 
                                    atkmodel_train=False, device=self.device)
                    #    train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)                
                       # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")
                    
                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(honest_norm_diff)
            if self.defense_technique == "foolsgold":
                delta = np.zeros((self.num_nets, pytorch_total_params))
                if memory_size > 0:
                    for net_idx, global_client_indx in enumerate(selected_node_indices):
                        flatten_local_model = flatten_model(net_list[net_idx])
                        local_update = flatten_local_model - self.flatten_net_avg
                        delta[global_client_indx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[global_client_indx, :]) > 1:
                            delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(delta[global_client_indx, :])

                        delta_memory[global_client_indx, :, flr % memory_size] = delta[global_client_indx, :]
                    # Track the total vector from each individual client
                    summed_deltas = np.sum(delta_memory, axis=2)      
                else:
                    for net_idx, global_client_indx in enumerate(selected_node_indices):
                        flatten_local_model = flatten_model(net_list[net_idx])
                        local_update = flatten_local_model - self.flatten_net_avg
                        local_update = local_update.detach().cpu().numpy()
                        delta[global_client_indx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[global_client_indx, :]) > 1:
                            delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(delta[global_client_indx, :])
                    # Track the total vector from each individual client
                    # print(f"delta={delta[selected_node_indices,:]}")
                    # print(f"summed_deltas[selected_node_indices,:].shape is: {summed_deltas[selected_node_indices,:].shape}")

                    summed_deltas[selected_node_indices,:] = summed_deltas[selected_node_indices,:] + delta[selected_node_indices,:]
                    # print(f"summed_deltas.shape is: {summed_deltas.shape}")
                    # print(f"summed_deltas={summed_deltas[selected_node_indices,:]}")
 
            ### conduct defense here:
            if self.defense_technique == "no-defense":
                pass
            elif self.defense_technique == "norm-clipping":
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "norm-clipping-adaptive":
                # we will need to adapt the norm diff first before the norm diff clipping
                logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector, 
                    np.mean(norm_diff_collector)))
                self._defender.norm_bound = np.mean(norm_diff_collector)
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "weak-dp":
                # this guy is just going to clip norm. No noise added here
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net,
                                        global_model=self.net_avg,)
            elif self.defense_technique == "krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=num_data_points,
                                                        g_user_indices=selected_node_indices,
                                                        device=self.device)
            elif self.defense_technique == "multi-krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=num_data_pointsos,
                                                        g_user_indices=selected_node_indices,
                                                        device=self.device)
            elif self.defense_technique == 'rlr':
                print(f"num_data_points: {num_data_points}")
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        num_dps=num_data_points,
                                                        global_model=self.net_avg)
            elif self.defense_technique == "foolsgold":
                print(f"Performing foolsgold...")
                net_list, net_freq = self._defender.exec(client_models=net_list, delta = delta[selected_node_indices,:] ,summed_deltas=summed_deltas[selected_node_indices,:], net_avg=self.net_avg, r=flr, device=self.device)
            elif self.defense_technique == "rfa":
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        net_freq=net_freq,
                                                        maxiter=500,
                                                        eps=1e-5,
                                                        ftol=1e-7,
                                                        device=self.device)
            else:
                NotImplementedError("Unsupported defense method !")

            # aggregation of the attack models distributed among participants
            atk_model_freq = [1.0/total_attackers for _ in range(total_attackers)]
            if total_attackers:
                # atk_vectorized_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in list(atk_model_list.values())]
                # avg_atk_net_vec = np.average(atk_vectorized_nets, weights=atk_model_freq, axis=0).astype(np.float32)
                # load_model_weight(self.atkmodel, torch.from_numpy(avg_atk_net_vec.astype(np.float32)).to(self.device))
                self.atkmodel = fed_avg_aggregator(self.atkmodel, list(atk_model_list.values()), atk_model_freq, device=self.device, model="atkmodel")
                self.tgtmodel.load_state_dict(self.atkmodel.state_dict())
            # testing for aggregated attack model
            local_acc_clean, local_acc_poison = test_updated(self.lira_args, self.device, self.tgtmodel, net, target_transform, 
                self.clean_train_loader, self.vanilla_emnist_test_loader, e, self.lira_args['train_epoch'], clip_image,
                log_prefix='Internal', dataset=self.dataset, criterion=criterion, subpath_saved=subpath_trigger_saved, test_eps=cur_training_eps)
            # after local training periods
            
            fed_aggregator = self.aggregator
            print(f"\nStart performing fed aggregator: {fed_aggregator}...")
            fed_nova_rho = 0.1
            list_ai = [(tau - fed_nova_rho * (1 - pow(fed_nova_rho, tau)) / (1 - fed_nova_rho)) / (1 - fed_nova_rho) for tau in list_tau]
                
            if fed_aggregator == "fednova":
                self.net_avg = fed_nova_aggregator(self.net_avg, net_list, list_ni, device, list_ai)
            else:
                self.net_avg = fed_avg_aggregator(self.net_avg, net_list, net_freq, device=self.device, model=self.model)
            
            self.flatten_net_avg = flatten_model(self.net_avg)
            #overall_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            #backdoor_acc = test(self.net_avg, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
            # overall_acc, raw_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            # backdoor_acc, _ = test(self.net_avg, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
 
            """GLOBAL TESTING FOR LIRA"""
            overall_acc, raw_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, 
                                        criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            

            acc_clean, backdoor_acc = test_updated(args=self.lira_args, device=self.device, atkmodel=self.tgtmodel, model=self.net_avg, target_transform=target_transform, 
                                    train_loader=self.clean_train_loader, test_loader=self.vanilla_emnist_test_loader,epoch=flr, trainepoch=self.lira_args['train_epoch'], 
                                    clip_image=clip_image, testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=criterion, 
                                    subpath_saved=subpath_trigger_saved, test_eps=cur_training_eps)

            if self.defense_technique == "weak-dp":
                # add noise to self.net_avg
                noise_adder = AddNoise(stddev=self.stddev)
                noise_adder.exec(client_model=self.net_avg,
                                                device=self.device)

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.net_avg, epoch=0, fl_round=flr, mode="avg")
            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            if overall_acc > best_main_acc:
                best_main_acc = overall_acc
                saved_path = f"checkpoint/bestmodel"
                atk_saved_path = f"checkpoint/atkmodel"
                
                torch.save(self.net_avg.state_dict(), f"{saved_path}/{self.dataset}_{self.model}.pt")
                torch.save(self.net_avg.state_dict(), f"{atk_saved_path}/{self.dataset}_{self.model}_{self.lira_args['attack_model']}.pt")
            
            if acc_clean > best_acc_clean or (acc_clean+self.lira_args['best_threshold'] > best_acc_clean and best_acc_poison < backdoor_acc) and self.retrain:
                best_acc_poison = backdoor_acc
                best_acc_clean = acc_clean
                print(f"Saving atk model with: backdoor_acc_{backdoor_acc} and best_acc_clean_{best_acc_clean}")
                torch.save({'atkmodel': self.atkmodel.state_dict(), 'clsmodel': self.net_avg.state_dict(), 
                            'best_acc_poison': best_acc_poison, 'best_acc_clean': best_acc_clean}, 
                           bestmodel_path)
            
            wandb_logging_items = {
                'fl_iter': flr,
                'main_task_acc': acc_clean*100.0,
                'backdoor_task_acc': backdoor_acc*100.0, 
                'local_best_acc_clean': best_acc_clean,
                'local_best_acc_poison': best_acc_poison,
                'local_MA': local_acc_clean*100.0,
                'local_BA': local_acc_poison*100.0
            }
            if wandb_ins:
                print(f"start logging to wandb")
                wandb_ins.log({"General Information": wandb_logging_items})

            raw_acc = 0
            overall_acc = acc_clean                
            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            raw_task_acc.append(raw_acc)
            backdoor_task_acc.append(backdoor_acc)
            if len(current_adv_norm_diff_list) == 0:
                adv_norm_diff_list.append(0)
            else:
                # if you have multiple adversaries in a round, average their norm diff
                adv_norm_diff_list.append(1.0*sum(current_adv_norm_diff_list)/len(current_adv_norm_diff_list))
        
        df = pd.DataFrame({'fl_iter': fl_iter_list, 
                            'main_task_acc': main_task_acc, 
                            'backdoor_acc': backdoor_task_acc, 
                            'raw_task_acc':raw_task_acc, 
                            'adv_norm_diff': adv_norm_diff_list, 
                            'wg_norm': wg_norm_list
                            })
       
        if self.poison_type == 'ardis':
            # add a row showing initial accuracies
            df1 = pd.DataFrame({'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]})
            df = pd.concat([df1, df])

        results_filename = get_results_filename(self.poison_type, self.attack_method, self.model_replacement, self.project_frequency,
                self.defense_technique, self.norm_bound, self.prox_attack, fixed_pool=True, model_arch=self.model)
        df.to_csv(results_filename, index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))
