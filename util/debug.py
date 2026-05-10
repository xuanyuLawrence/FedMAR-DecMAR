import torch

def check_if_client_update(prev_state_dict, cur_state_dict):
    diff = 0
    for param_tensor in prev_state_dict:
        t0 = prev_state_dict[param_tensor]
        t1 = cur_state_dict[param_tensor]
        if t0.is_cuda:
            if t0.requires_grad:
                diff += torch.sum(t1 - t0).detach().cpu().item()
            else:
                diff += torch.sum(t1 - t0).cpu().item()
        else:
            if t0.requires_grad:
                diff += torch.sum(t1 - t0).detach().item()
            else:
                diff += torch.sum(t1 - t0).item()
    if diff != 0:
        return True
    return False
