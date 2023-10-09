import torch
from utils import *

from geometric_median import geometric_median

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()

def rlr_avg(vectorize_nets, vectorize_avg_net, freq, attacker_idxs, lr, n_params, device, robustLR_threshold=4):
    lr_vector = torch.Tensor([lr]*n_params).to(device)
    total_client = len(vectorize_nets)
    local_updates = vectorize_nets - vectorize_avg_net
    print(f"len freq: {len(freq)}")
    print(f"local_updates.shape is: {len(local_updates)}")
    fed_avg_updates_vector = np.average(local_updates, weights=freq, axis=0).astype(float32)
    print(f"fed_avg_vector.shape is: {fed_avg_updates_vector.shape}")
    # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
    selected_net_indx = [i for i in range(total_client) if i not in attacker_idxs]
    selected_freq = np.array(freq)[selected_net_indx]
    selected_freq = [freq/sum(selected_freq) for freq in selected_freq]
    
    
    agent_updates_sign = [np.sign(update) for update in local_updates]  
    sm_of_signs = np.abs(sum(agent_updates_sign))
    sm_of_signs[sm_of_signs < robustLR_threshold] = -lr
    sm_of_signs[sm_of_signs >= robustLR_threshold] = lr
    print(f"sm_of_signs is: {sm_of_signs}")
    
    lr_vector = sm_of_signs
    poison_w_idxs = sm_of_signs < 0
    # poison_w_idxs = poison_w_idxs*1
    print(f"poison_w_idxs: {poison_w_idxs}")
    print(f"lr_vector: {lr_vector}")
    local_updates = np.asarray(local_updates)
    print(f"local_updates.shape is: {local_updates.shape}")
    # local_updates[attacker_idxs][poison_w_idxs] = 0
    cnt = 0
    sm_updates_2 = 0
    # for _id, update in enumerate(local_updates):
    #     if _id not in attacker_idxs:
    #         sm_updates_2 += selected_freq[cnt]*update[poison_w_idxs]
    #         cnt+=1
    for _id, update in enumerate(local_updates):
        if _id not in attacker_idxs:
            sm_updates_2 += freq[_id]*update[poison_w_idxs]
        else:
            sm_updates_2 += freq[_id]*(-update[poison_w_idxs])
            
    print(f"sm_updates_2.shape is: {sm_updates_2.shape}")
    fed_avg_updates_vector[poison_w_idxs] = sm_updates_2
    new_global_params =  (vectorize_avg_net + lr*fed_avg_updates_vector).astype(np.float32)
    return new_global_params

class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class ClippingDefense(Defense):
    """
    Deprecated, do not use this method
    """
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, *args, **kwargs):
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net/max(1, weight_norm/self.norm_bound)

        logger.info("Norm Clipped Mode {}".format(
            torch.norm(clipped_weight).item()))
        load_model_weight(client_model, clipped_weight)        
        # index_bias = 0
        # for p_index, p in enumerate(client_model.parameters()):
        #     p.data =  clipped_weight[index_bias:index_bias+p.numel()].view(p.size())
        #     index_bias += p.numel()
        ##weight_norm = torch.sqrt(sum([torch.norm(p)**2 for p in client_model.parameters()]))
        #for p_index, p in enumerate(client_model.parameters()):
        #    p.data /= max(1, weight_norm/self.norm_bound)
        return None

class RLR(Defense):
    def __init__(self, n_params, device, args, agent_data_sizes=[], writer=None, robustLR_threshold = 0, aggr="avg", poisoned_val_loader=None):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        # print(f"args: {args}")
        # self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = None
        self.cum_net_mov = 0
        self.device = device
        self.robustLR_threshold = robustLR_threshold
        
         
    def exec(self, global_model, client_models, num_dps, agent_updates_dict=None, cur_round=0):
        # adjust LR if robust LR is selected
        print(f"self.args: {self.args}")
        print(f"self.args['server_lr']: {self.args['server_lr']}")
        lr_vector = torch.Tensor([self.args['server_lr']]*self.n_params).to(self.device)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(global_model).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net
        aggr_freq = [num_dp/sum(num_dps) for num_dp in num_dps]
        
        if self.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(local_updates)
        
        
        aggregated_updates = 0
        if self.args['aggr']=='avg':          
            aggregated_updates = self.agg_avg(local_updates, aggr_freq)
        elif self.args['aggr']=='comed':
            #TODO update for the 2 remaining func
            aggregated_updates = self.agg_comed(local_updates)
        elif self.args['aggr'] == 'sign':
            aggregated_updates = self.agg_sign(local_updates)
            
        if self.args['noise'] > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args['noise']*self.args['clip'], size=(self.n_params,)).to(self.device))

        cur_global_params = vectorize_avg_net
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).astype(np.float32)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(new_global_params).to(self.device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq
        
        
        # some plotting stuff if desired
        # self.plot_sign_agreement(lr_vector, cur_global_params, new_global_params, cur_round)
        # self.plot_norms(agent_updates_dict, cur_round)
     
    
    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [np.sign(update) for update in agent_updates]  
        sm_of_signs = np.abs(sum(agent_updates_sign))
        print(f"sm_of_signs is: {sm_of_signs}")
        
        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.args['server_lr']
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.args['server_lr']                                            
        return sm_of_signs
        
            
    def agg_avg(self, agent_updates_dict, num_dps):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in enumerate(agent_updates_dict):
            n_agent_data = num_dps[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    # def exec(self, global_model, client_models, num_dps, agent_updates_dict=None, cur_round=0):
    #     # Adjust LR if robust LR is selected
    #     print(f"self.args: {self.args}")
    #     print(f"self.args['server_lr']: {self.args['server_lr']}")
    #     lr_vector = torch.Tensor([self.args['server_lr']] * self.n_params).to(self.device)
    #     vectorize_nets = [vectorize_net(cm).detach() for cm in client_models]
    #     vectorize_avg_net = vectorize_net(global_model).detach()
    #     local_updates = [vn - vectorize_avg_net for vn in vectorize_nets]
    #     aggr_freq = [num_dp / sum(num_dps) for num_dp in num_dps]

    #     if self.robustLR_threshold > 0:
    #         lr_vector = self.compute_robustLR(local_updates)

    #     aggregated_updates = 0
    #     if self.args['aggr'] == 'avg':
    #         aggregated_updates = self.agg_avg(local_updates, num_dps)
    #     elif self.args['aggr'] == 'comed':
    #         # TODO update for the 2 remaining func
    #         aggregated_updates = self.agg_comed(local_updates)
    #     elif self.args['aggr'] == 'sign':
    #         aggregated_updates = self.agg_sign(local_updates)

    #     if self.args['noise'] > 0:
    #         noise = torch.normal(mean=0, std=self.args['noise'] * self.args['clip'], size=(self.n_params,)).to(self.device)
    #         aggregated_updates.add_(noise)

    #     cur_global_params = vectorize_avg_net.to(self.device)
    #     new_global_params = cur_global_params + lr_vector * aggregated_updates.float()

    #     aggregated_model = client_models[0]  # Slicing which doesn't really matter
    #     load_model_weight(aggregated_model, new_global_params)
    #     neo_net_list = [aggregated_model]
    #     neo_net_freq = [1.0]
    #     return neo_net_list, neo_net_freq


    # def compute_robustLR(self, agent_updates):
    #     agent_updates_sign = [torch.sign(update) for update in agent_updates]
    #     sm_of_signs = torch.abs(sum(agent_updates_sign))
    #     print(f"sm_of_signs is: {sm_of_signs}")

    #     sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.args['server_lr']
    #     sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.args['server_lr']
    #     return sm_of_signs


    # def agg_avg(self, agent_updates_dict, num_dps):
    #     """Classic federated average"""
    #     sm_updates, total_data = 0, 0
    #     for _id, update in enumerate(agent_updates_dict):
    #         n_agent_data = num_dps[_id]
    #         sm_updates += n_agent_data * update
    #         total_data += n_agent_data
    #     return sm_updates / total_data

    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args['clip']))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return
        
    def comp_diag_fisher(self, model_params, data_loader, adv=True):

        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
            
        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)
                
            outputs = model(inputs)
            log_all_probs = F.log_softmax(outputs, dim=1)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()
            
            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
                
        return parameters_to_vector(precision_matrices.values()).detach()

        
    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        """ Getting sign agreement of updates between honest and corrupt agents """
        # total update for this round
        update = new_global_params - cur_global_params
        
        # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
        fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
        fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
        _, adv_idxs = fisher_adv.sort()
        _, hon_idxs = fisher_hon.sort()
        
        # get most important n_idxs params
        n_idxs = self.args.top_frac #math.floor(self.n_params*self.args.top_frac)
        adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
        hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()
        
        # minimized and maximized indexes
        min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
        max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()
        
        # get minimized and maximized idxs for adversary and honest
        max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
        max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
        min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
        min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)
       
        # get differences
        max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
        max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
        min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
        min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)
        
        # get actual update values and compute L2 norm
        max_adv_only_upd = update[max_adv_only_idxs] # S1
        max_hon_only_upd = update[max_hon_only_idxs] # S2
        
        min_adv_only_upd = update[min_adv_only_idxs] # S3
        min_hon_only_upd = update[min_hon_only_idxs] # S4


        #log l2 of updates
        max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
        max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
        min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
        min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()
       
        self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)
        
        
        net_adv =  max_adv_only_upd_l2 - min_adv_only_upd_l2
        net_hon =  max_hon_only_upd_l2 - min_hon_only_upd_l2
        self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)
        
        self.cum_net_mov += (net_hon - net_adv)
        self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
        return

class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorize_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorize_diff).item()
        clipped_weight_diff = vectorize_diff/max(1, weight_diff_norm/self.norm_bound)

        logger.info("Norm Weight Diff: {}, Norm Clipped Weight Diff {}".format(weight_diff_norm,
            torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None

class WeakDPDefense(Defense):
    """
        deprecated: don't use!
        according to literature, DPDefense should be applied
        to the aggregated model, not invidual models
        """
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net/max(1, weight_norm/self.norm_bound)
        dp_weight = clipped_weight + torch.randn(
            vectorized_net.size(),device=self.device) * self.stddev

        load_model_weight(client_model, clipped_weight)
        return None

class AddNoise(Defense):
    def __init__(self, stddev, *args, **kwargs):
        self.stddev = stddev

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        gaussian_noise = torch.randn(vectorized_net.size(),
                            device=self.device) * self.stddev
        dp_weight = vectorized_net + gaussian_noise
        load_model_weight(client_model, dp_weight)
        logger.info("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))
        
        return None

class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps, g_user_indices, device, *args, **kwargs):
        # # print("Start performing krum")
        # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        
        # neighbor_distances = []
        # for i, g_i in enumerate(vectorize_nets):
        #     distance = []
        #     for j in range(i+1, len(vectorize_nets)):
        #         if i != j:
        #             g_j = vectorize_nets[j]
        #             distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
        #     neighbor_distances.append(distance)
        vectorize_nets = [vectorize_net(cm).detach() for cm in client_models]

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(torch.norm(g_i - g_j).pow(2).item())
            neighbor_distances.append(distance)

        # # compute scores
        # nb_in_score = self.num_workers-self.s-2
        # scores = []
        # for i, g_i in enumerate(vectorize_nets):
        #     dists = []
        #     for j, g_j in enumerate(vectorize_nets):
        #         if j == i:
        #             continue
        #         if j < i:
        #             dists.append(neighbor_distances[j][i - j - 1])
        #         else:
        #             dists.append(neighbor_distances[i][j - i - 1])
        #     # alternative to topk in pytorch and tensorflow
        #     topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
        #     scores.append(sum(np.take(dists, topk_ind)))
        
        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in PyTorch
            dists_tensor = torch.tensor(dists)
            topk_values, _ = torch.topk(dists_tensor, nb_in_score)
            scores.append(torch.sum(topk_values).item())
            
        # if self._mode == "krum":
        #     i_star = scores.index(min(scores))
        #     logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
        #     aggregated_model = client_models[0] # slicing which doesn't really matter
        #     load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
        #     neo_net_list = [aggregated_model]
        #     logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
        #     neo_net_freq = [1.0]
        #     return neo_net_list, neo_net_freq
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            aggregated_model = client_models[0]  # create a clone of the model
            aggregated_model.load_state_dict(vectorize_nets[i_star].to(device))
            neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        # elif self._mode == "multi-krum":
        #     topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]
            
        #     # we reconstruct the weighted averaging here:
        #     selected_num_dps = np.array(num_dps)[topk_ind]
        #     reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

        #     logger.info("Num data points: {}".format(num_dps))
        #     logger.info("Num selected data points: {}".format(selected_num_dps))
        #     logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))
        #     #aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0)
        #     aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq, axis=0).astype(np.float32)

        #     aggregated_model = client_models[0] # slicing which doesn't really matter
        #     load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        #     neo_net_list = [aggregated_model]
        #     #logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
        #     neo_net_freq = [1.0]
        #     return neo_net_list, neo_net_freq
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]

            # We reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = torch.tensor([snd/sum(selected_num_dps) for snd in selected_num_dps], dtype=torch.float32, device=device)

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))
            
            aggregated_grad = torch.sum(torch.stack([reconstructed_freq[i] * vectorize_nets[j] for i, j in enumerate(topk_ind)], dim=0), dim=0)  # Weighted sum of the gradients
            
            aggregated_model = client_models[0]  # create a clone of the model
            load_model_weight(aggregated_model, aggregated_grad)
            neo_net_list = [aggregated_model]
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq


class CRFL(Defense):
    """
    we implement the robust aggregator of CRFL
    """
    TYPE_LOAN='loan'
    TYPE_MNIST='mnist'
    TYPE_EMNIST='emnist'
    TYPE_CIFAR10 = 'cifar10'
    TYPE_TINY_IMAGENET = 'tiny-imagenet'

    def __init__(self, *args, **kwargs):
        pass

    # def clip_weight_norm(self, model, thres):
    #     current_norm = 0
    #     for p in model.parameters():
    #         param_norm = p.data.norm(2)
    #         current_norm += param_norm.item() ** 2
    #         p.data = p.data * thres / max(param_norm, thres)
    #     current_norm = current_norm ** (1. / 2)
    #     return current_norm
    def model_global_norm(self, model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)
    
    def clip_weight_norm(self, model, clip):
        total_norm = self.model_global_norm(model)
        print("total_norm: " + str(total_norm)+ "clip_norm: "+str(clip ))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in model.named_parameters():
                layer.data.mul_(clip_coef)
            current_norm = self.model_global_norm(model)
            print("clip~~~ norm after clipping: "+ str(current_norm) )
        return current_norm

    def dp_noise(self, param, sigma):
        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
        return noised_layer
    
    def exec(self, target_model, epoch, sigma_param, dataset_name, device):
        # target_model: global model
        # clip the global model
        
        param_clip_thres = 0
        # sigma_param = 0.01
        if dataset_name == CRFL.TYPE_MNIST:
            dynamic_thres= epoch *0.1+2
            param_clip_thres = 15
        elif dataset_name == CRFL.TYPE_LOAN:
            dynamic_thres = epoch*0.025+2
            param_clip_thres = 5
        elif dataset_name == CRFL.TYPE_EMNIST:
            dynamic_thres= epoch*0.25+4
            param_clip_thres = 100
        # tuannm threshold by tuannm day 11.02.2023
        elif dataset_name == CRFL.TYPE_CIFAR10:
            dynamic_thres= epoch*0.25+4
            param_clip_thres = 100
        elif dataset_name == CRFL.TYPE_TINY_IMAGENET:
            dynamic_thres= epoch*0.25+4
            param_clip_thres = 100
           
            
        if dynamic_thres < param_clip_thres: 
            param_clip_thres= dynamic_thres
            
        current_norm = self.clip_weight_norm(target_model, param_clip_thres)
        print(f" epoch: {epoch} clip the global model current_norm: {current_norm} !")
        
        # add noise
        print(f" epoch: {epoch} add noise on the global model!")
        for name, param in target_model.state_dict().items():
            param.add_(self.dp_noise(param, sigma_param))
        
        return [target_model], [1.0]

# class RFA(Defense):
#     """
#     we implement the robust aggregator at: 
#     https://arxiv.org/pdf/1912.13445.pdf
#     the code is translated from the TensorFlow implementation: 
#     https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
#     """
#     def __init__(self, *args, **kwargs):
#         pass

#     def exec(self, client_models, net_freq, 
#                    maxiter=4, eps=1e-5,
#                    ftol=1e-6, device=torch.device("cuda"), 
#                     *args, **kwargs):
#         """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
#         """
#         # so alphas will be the same as the net freq in our code
#         alphas = np.asarray(net_freq, dtype=np.float32)
#         vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
#         median = self.weighted_average_oracle(vectorize_nets, alphas)

#         num_oracle_calls = 1

#         # logging
#         obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

#         logs = []
#         log_entry = [0, obj_val, 0, 0]
#         logs.append("Tracking log entry: {}".format(log_entry))
#         logger.info('Starting Weiszfeld algorithm')
#         logger.info(log_entry)

#         # start
#         for i in range(maxiter):
#             prev_median, prev_obj_val = median, obj_val
#             weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
#                                  dtype=alphas.dtype)
#             weights = weights / weights.sum()
#             median = self.weighted_average_oracle(vectorize_nets, weights)
#             num_oracle_calls += 1
#             obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
#             log_entry = [i+1, obj_val,
#                          (prev_obj_val - obj_val)/obj_val,
#                          self.l2dist(median, prev_median)]
#             logs.append(log_entry)
#             logs.append("Tracking log entry: {}".format(log_entry))
#             logger.info("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
#             if abs(prev_obj_val - obj_val) < ftol * obj_val:
#                 break
#         #logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

#         aggregated_model = client_models[0] # slicing which doesn't really matter
#         load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
#         neo_net_list = [aggregated_model]
#         neo_net_freq = [1.0]
#         return neo_net_list, neo_net_freq

#     def weighted_average_oracle(self, points, weights):
#         """Computes weighted average of atoms with specified weights
#         Args:
#             points: list, whose weighted average we wish to calculate
#                 Each element is a list_of_np.ndarray
#             weights: list of weights of the same length as atoms
#         """
#         ### original implementation in TFF
#         #tot_weights = np.sum(weights)
#         #weighted_updates = [np.zeros_like(v) for v in points[0]]
#         #for w, p in zip(weights, points):
#         #    for j, weighted_val in enumerate(weighted_updates):
#         #        weighted_val += (w / tot_weights) * p[j]
#         #return weighted_updates
#         ####
#         tot_weights = np.sum(weights)
#         weighted_updates = np.zeros(points[0].shape)
#         for w, p in zip(weights, points):
#             weighted_updates += (w * p / tot_weights)
#         return weighted_updates

#     def l2dist(self, p1, p2):
#         """L2 distance between p1, p2, each of which is a list of nd-arrays"""
#         # this is a helper function
#         return np.linalg.norm(p1 - p2)

#     def geometric_median_objective(self, median, points, alphas):
#         """Compute geometric median objective."""
#         return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])

class RFA(Defense):
    """
    we implement the robust aggregator at: 
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation: 
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq,
             maxiter=4, eps=1e-5,
             ftol=1e-6, device=torch.device("cuda"),
             *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        alphas = torch.tensor(net_freq, dtype=torch.float32, device=device)
        vectorize_nets = [vectorize_net(cm).detach() for cm in client_models]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        logger.info('Starting Weiszfeld algorithm')
        logger.info(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                   dtype=alphas.dtype, device=device)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            logger.info("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        #logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_models[0]  # create a clone of the model
        load_model_weight(aggregated_model, median.to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_torch.Tensor
            weights: list of weights of the same length as atoms
        """
        tot_weights = weights.sum()
        weighted_updates = torch.zeros(points[0].shape, dtype=points[0].dtype, device=points[0].device)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return torch.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return torch.sum(torch.stack([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)]))
class GeoMedian(Defense):
    """
    we implement the robust aggregator of Geometric Median (GM)
    """
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq, 
                   maxiter=4, eps=1e-5,
                   ftol=1e-6, device=torch.device("cuda"), 
                    *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = np.array([vectorize_net(cm).detach().cpu().numpy() for cm in client_models]).astype(np.float32)
        median = geometric_median(vectorize_nets)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

class FoolsGold(Defense):
    def __init__(self, num_clients, num_features, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clients = num_clients
        self.n_features = num_features
        self.n_classes = num_classes

    def get_cos_similarity(self, full_deltas):
        '''
        Returns the pairwise cosine similarity of client gradients
        '''
        if True in np.isnan(full_deltas):
            pdb.set_trace()
        return smp.cosine_similarity(full_deltas)

    def importanceFeatureMapGlobal(self, model):
        # aggregate = np.abs(np.sum( np.reshape(model, (10, 784)), axis=0))
        # aggregate = aggregate / np.linalg.norm(aggregate)
        # return np.repeat(aggregate, 10)
        return np.abs(model) / np.sum(np.abs(model))

    def importanceFeatureMapLocal(self, model, topk_prop=0.5):
        # model: np arr
        d = self.n_features # dim of flatten weight
        class_d = int(d / self.n_classes)

        M = model.copy()
        M = np.reshape(M, (self.n_classes, class_d))
        
        # #Take abs?
        # M = np.abs(M)

        for i in range(self.n_classes):
            if (M[i].sum() == 0):
                pdb.set_trace()
            M[i] = np.abs(M[i] - M[i].mean())
            
            M[i] = M[i] / M[i].sum()

            # Top k of 784
            topk = int(class_d * topk_prop)
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            M[i][sig_features_idx] = 0
        
        return M.flatten()   

    def importanceFeatureHard(self, model, topk_prop=0.5):

        class_d = int(self.n_features / self.n_classes)

        M = np.reshape(model, (self.n_classes, class_d))
        importantFeatures = np.ones((self.n_classes, class_d))
        # Top k of 784
        topk = int(class_d * topk_prop)
        for i in range(self.n_classes):
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]     
            importantFeatures[i][sig_features_idx] = 0
        return importantFeatures.flatten()  

    def get_krum_scores(self, X, groupsize):

        krum_scores = np.zeros(len(X))

        # Calculate distances
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(
            X**2, axis=1)[None] - 2 * np.dot(X, X.T)

        for i in range(len(X)):
            krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

        return krum_scores

    def foolsgold(self, this_delta, summed_deltas, sig_features_idx, iter, model, topk_prop=0, importance=False, importanceHard=False, clip=0):
        epsilon = 1e-5
        # Take all the features of sig_features_idx for each clients
        sd = summed_deltas.copy()
        sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

        if importance or importanceHard:
            if importance:
                # smooth version of importance features
                importantFeatures = self.importanceFeatureMapLocal(model, topk_prop)
            if importanceHard:
                # hard version of important features
                importantFeatures = self.importanceFeatureHard(model, topk_prop)
            for i in range(self.n_clients):
                sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)
                
        N, _ = sig_filtered_deltas.shape
        cs = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    cs[i,i] = 1  
                    continue
                if cs[i,j] != 0 and cs[j,i] != 0:
                    continue
                dot_i = sig_filtered_deltas[i][np.newaxis, :] @ sig_filtered_deltas[j][:, np.newaxis]
                norm_mul = np.linalg.norm(sig_filtered_deltas[i]) * np.linalg.norm(sig_filtered_deltas[j])
                cs[i, j] = cs[j, i] = dot_i / norm_mul
        
        cs = cs - np.eye(N)
        # Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)

        wv[(wv == 1)] = .99
        
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        if clip != 0:

            # Augment onto krum
            scores = self.get_krum_scores(this_delta, self.n_clients - clip)
            bad_idx = np.argpartition(scores, self.n_clients - clip)[(self.n_clients - clip):self.n_clients]

            # Filter out the highest krum scores
            wv[bad_idx] = 0

        print(f"wv: {wv}")
        wv = wv/sum(wv)
        avg_updates = np.average(this_delta, axis=0, weights=wv)
        return avg_updates, wv

    def exec(self, client_models, delta, summed_deltas, net_avg, r, device, *args, **kwargs):
        '''
        Aggregates history of gradient directions
        '''
        print(f"START Aggregating history of gradient directions")
        # total_client = len(client_models)
        # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(net_avg).detach().cpu().numpy()
        flatten_net_avg = vectorize_net(net_avg).detach().cpu().numpy()

        # Significant features filter, the top k biggest weights
        topk = int(self.n_features / 2)
        sig_features_idx = np.argpartition(flatten_net_avg, -topk)[-topk:]
        sig_features_idx = np.arange(self.n_features)
        avg_delta, wv = self.foolsgold(delta, summed_deltas, sig_features_idx, r, vectorize_avg_net, clip = 0)
        return wv

if __name__ == "__main__":
    # some tests here
    import copy
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

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
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check 1, this should recover the global model
    sim_global_model = Net().to(device)
    sim_local_model1 = copy.deepcopy(sim_global_model)
    #sim_local_model = Net().to(device)
    defender = WeightDiffClippingDefense(norm_bound=5)
    defender.exec(client_model=sim_local_model1, global_model=sim_global_model)

    vec_global_sim_net = vectorize_net(sim_global_model)
    vec_local_sim_net1 = vectorize_net(sim_local_model1)

    # Norm Weight Diff: 0.0, Norm Clipped Weight Diff 0.0
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 8.843663215637207    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(), 
        torch.norm(vec_local_sim_net1).item()))

    # check 2, adding some large perturbation
    sim_local_model2 = copy.deepcopy(sim_global_model)
    scaling_facor = 2
    for p_index, p in enumerate(sim_local_model2.parameters()):
        p.data = p.data + torch.randn(p.size()) * scaling_facor
    defender.exec(client_model=sim_local_model2, global_model=sim_global_model)
    vec_local_sim_net2 = vectorize_net(sim_local_model2)

    # Norm Weight Diff: 2191.04345703125, Norm Clipped Weight Diff 4.999983787536621
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 10.155366897583008    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(), 
        torch.norm(vec_local_sim_net2).item()))
