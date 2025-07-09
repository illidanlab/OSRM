import os
from tqdm import tqdm
from copy import deepcopy
import torch
from typing import Dict, Tuple
from torch import nn

def get_post_analytical_init_weights(model:nn.Module, ptr_model:nn.Module, right_lora_init_weights:dict, target_modules:Tuple[list]):
    left_lora_init_weights = {}
    for (param_name, param), (ptr_param_name, ptr_param) in zip(model.named_modules(), ptr_model.named_modules()):
        for target_module in target_modules:
            if f".{target_module}" in param_name:
                BA = param.weight - ptr_param.weight
                A_pinv_analytical = torch.linalg.pinv(right_lora_init_weights['base_model.model.'+param_name+'.lora_A.default.weight']).to(torch.bfloat16)
                rec_B = torch.matmul(BA.to(f'cuda:{torch.cuda.device_count()-1}'), A_pinv_analytical.to(f'cuda:{torch.cuda.device_count()-1}'))
                left_lora_init_weights['base_model.model.'+param_name+'.lora_B.default.weight'] = rec_B.to(torch.bfloat16)
    return left_lora_init_weights

def get_pretrained_latent_features(model:nn.Module, rank:int, dataloaders:dict, target_modules:list):
    '''get the latent features of the pretrained model for all tasks and store them:
        :dataloaders : a dictionary of dataloaders for all tasks
    '''
    model.eval()
    all_task_latent_features = {}

    def get_hook(module_name: str):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            x = input[0].detach() # x: [batch_size, seq_len, hidden_size]
            batch_num_actual_examples = x.shape[0]            
            '''analytical v2:
                X = [x_1; x_2; ...; x_n], where n is the #tasks, and x_i is of shape [hidden_size]
            '''
            x = x.mean(dim=1).mean(dim=0).to(f'cuda:{torch.cuda.device_count()-1}') # x: [hidden_size]
            tmp_right_lora_init_weights['base_model.model.'+module_name+'.lora_A.default.weight'] = tmp_right_lora_init_weights['base_model.model.'+module_name+'.lora_A.default.weight'] + x
        return hook

    model = model.to(torch.bfloat16)
    for task_name, dataloader in dataloaders.items():
        right_lora_init_weights = {}
        # register hooks
        handles = []
        tmp_right_lora_init_weights = {}
        for name, param in model.named_modules():
            for target_module in target_modules:
                if f".{target_module}" in name:
                    tmp_right_lora_init_weights['base_model.model.'+name+'.lora_A.default.weight'] = torch.zeros(param.weight.shape[1]).to(f'cuda:{torch.cuda.device_count()-1}')
                    handle = param.register_forward_hook(get_hook(name))
                    handles.append(handle)
        
        # forward pass:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {task_name}")):
            with torch.no_grad():
                # batch = batch.to(f'cuda')
                outputs = model(**batch)

        # remove the added hook
        for handle in handles:
            handle.remove()
        
        # cat the data matrix of each task:
        for name, weight in tmp_right_lora_init_weights.items():
            avg_weight = weight / batch_idx
            avg_weight = avg_weight.unsqueeze(0)
            right_lora_init_weights[name] = torch.cat((right_lora_init_weights.get(name, torch.tensor([]).to(f'cuda:{torch.cuda.device_count()-1}')), avg_weight), dim=0)
        
        all_task_latent_features[task_name] = right_lora_init_weights
    
    return all_task_latent_features

def peft_analytical_init(rank:int, external_task_latent_features:dict):
    '''Reduce running time based on PEFT analytical initialization V2:'''
    right_lora_init_weights = {}
    for task_name, latent_features in external_task_latent_features.items():
        # cat the data matrix of each task:
        for name, weight in latent_features.items():
            right_lora_init_weights[name] = torch.cat((right_lora_init_weights.get(name, torch.tensor([]).to(f'cuda:{torch.cuda.device_count()-1}')), weight), dim=0)
    
    # compute the null space of X
    for name, weight in right_lora_init_weights.items():
        '''below is the code for comuting the null space of X:'''
        # _weight = deepcopy(weight.clone())
        U, S, Vt = torch.linalg.svd(weight)
        V = Vt.T
        right_lora_init_weights[name] = V[:, -rank:].T
        # print(torch.norm(torch.matmul(_weight, right_lora_init_weights[name].T)))

        '''below is the code for the constraint of AA^T=I:'''
        # XtX = torch.matmul(weight.T, weight)
        # eigvals, eigvecs = torch.linalg.eigh(XtX)
        # right_lora_init_weights[name] = eigvecs.T[-rank:, :]

    return right_lora_init_weights