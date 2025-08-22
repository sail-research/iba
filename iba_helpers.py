import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Constants for ImageNet normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN = ((np.array([0, 0, 0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX = ((np.array([1, 1, 1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()

# Global loss function
loss_fn = nn.CrossEntropyLoss()


def get_clip_image(dataset="cifar10"):
    """Get image clipping function based on dataset.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        function: Clipping function for the specified dataset
    """
    if dataset in ['tiny-imagenet', 'tiny-imagenet32', 'cifar10', 'gtsrb']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'mnist':
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    return clip_image


def flatten_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer.
    
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    
    Args:
        tensors (Iterable[Tensor]): Dense tensors to flatten
        
    Returns:
        Tensor: A 1D buffer containing input tensors
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def flatten_model(model):
    """Flatten all model parameters into a single tensor.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tensor: Flattened model parameters
    """
    return torch.cat([flatten_tensors(i) for i in model.parameters()])


def all2one_target_transform(x, attack_target=1):
    """Transform all targets to a single target class.
    
    Args:
        x: Input tensor
        attack_target (int): Target class for all inputs
        
    Returns:
        Tensor: Transformed targets
    """
    return torch.ones_like(x) * attack_target


def all2all_target_transform(x, num_classes):
    """Transform targets using all2all mapping.
    
    Args:
        x: Input tensor
        num_classes (int): Number of classes
        
    Returns:
        Tensor: Transformed targets
    """
    return (x + 1) % num_classes


def get_target_transform(args):
    """Get target transform function based on attack mode.
    
    Args:
        args (dict): Arguments containing mode and other parameters
        
    Returns:
        function: Target transform function
    """
    if args is None:
        raise ValueError("args cannot be None in get_target_transform")
    
    if args['mode'] == 'all2one':
        target_transform = lambda x: all2one_target_transform(x, args['target_label'])
    elif args['mode'] == 'all2all':
        target_transform = lambda x: all2all_target_transform(x, args['num_classes'])
    else:
        raise ValueError(f'Invalid mode {args["mode"]}')
    return target_transform


def create_trigger_model(dataset, device="cpu", attack_model=None):
    """Create trigger model for the specified dataset.
    
    Args:
        dataset (str): Dataset name
        device (str): Device to place models on
        attack_model (str, optional): Type of attack model
        
    Returns:
        tuple: (attack_model, target_model)
    """
    if dataset == 'cifar10':
        from attack_models.unet import UNet
        atkmodel = UNet(3).to(device)
        tgtmodel = UNet(3).to(device)
    elif dataset == 'mnist':
        from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
        atkmodel = Autoencoder().to(device)
        tgtmodel = Autoencoder().to(device)
    elif dataset in ['tiny-imagenet', 'tiny-imagenet32', 'gtsrb']:
        if attack_model is None or attack_model == 'autoencoder':
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder().to(device)
            tgtmodel = Autoencoder().to(device)
        elif attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(device)
            tgtmodel = UNet(3).to(device)
        else:
            raise ValueError(f'Invalid attack model: {attack_model}')
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    
    return atkmodel, tgtmodel

def init_model(model, device):
    """Initialize model parameters.
    
    Args:
        model: PyTorch model
        device: Device to place model on
    """
    for param in model.parameters():
        param.data.fill_(0)
    return model


def create_paths(args):
    """Create paths for model saving and loading.
    
    Args:
        args (dict): Arguments containing path configuration
        
    Returns:
        tuple: (base_path, checkpoint_path, best_model_path)
    """
    if args is None:
        raise ValueError("args cannot be None in create_paths")
    
    if args['mode'] == 'all2one': 
        basepath = os.path.join(args['path'], f"{args['mode']}_{args['target_label']}", 
                               args['dataset'], args['clsmodel'])
    else:
        basepath = os.path.join(args['path'], args['mode'], args['dataset'], args['clsmodel'])
   
    basepath = os.path.join(basepath, 
                           f"lr{args['lr']}-lratk{args['lr_atk']}-eps{args['eps']}-alpha{args['attack_alpha']}-clsepoch{args['train_epoch']}-atkmodel{args['attack_model']}-atk{args['attack_portion']}")

    if not os.path.exists(basepath):
        print(f'Creating new model training in {basepath}')
        os.makedirs(basepath)
        
    checkpoint_path = os.path.join(basepath, 'checkpoint.ckpt')
    bestmodel_path = os.path.join(basepath, 'bestmodel.ckpt')
    return basepath, checkpoint_path, bestmodel_path


def vectorize_net(net):
    """Vectorize network parameters.
    
    Args:
        net: PyTorch model
        
    Returns:
        Tensor: Vectorized network parameters
    """
    return torch.cat([p.view(-1) for p in net.parameters()])


def get_grad_mask_F3BA(model_vec, ori_model_vec, optimizer=None, clean_dataloader=None, ratio=0.01, device="cpu"):
    """Generate gradient mask using F3BA method.
    
    Args:
        model_vec: Vectorized current model
        ori_model_vec: Vectorized original model
        optimizer: Optimizer (unused)
        clean_dataloader: Clean dataloader (unused)
        ratio (float): Ratio of parameters to mask
        device (str): Device (unused)
        
    Returns:
        Tensor: Gradient mask
    """
    important_scores = (model_vec - ori_model_vec).mul(ori_model_vec)
    _, indices = torch.topk(-1 * important_scores, int(len(important_scores) * ratio))
    mask_flat_all_layer = torch.zeros(len(important_scores))
    mask_flat_all_layer[indices] = 1.0
    return mask_flat_all_layer


def apply_grad_mask(model, mask_grad_list):
    """Apply gradient mask to model parameters.
    
    Args:
        model: PyTorch model
        mask_grad_list: List of gradient masks
    """
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            try:
                mask = next(mask_grad_list_copy)
                parms.grad.data.mul_(mask)
            except StopIteration:
                break


def compute_hessian(model, loss):
    """Compute Hessian matrix for the model parameters.
    
    Args:
        model: PyTorch model
        loss: Loss tensor
        
    Returns:
        list: List of Hessian matrices for each layer
    """
    H = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grad_params = torch.autograd.grad(loss.mean(), parameter, create_graph=True)
            length_grad = len(grad_params[0])
            hessian = torch.zeros(length_grad, length_grad)
            
            for parameter_loc in range(len(parameter)):
                if parameter[parameter_loc].data.cpu().numpy().sum() == 0.0:
                    continue
                grad_params2 = torch.autograd.grad(grad_params[0][parameter_loc].mean(), 
                                                 parameter, create_graph=True)
                hessian[parameter_loc, :] = grad_params2[0].data
            
            H.append(hessian)
    
    return H
    
