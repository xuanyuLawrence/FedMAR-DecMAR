import torch
import copy
import random
import util.misc as misc
import os
from debug import ckpt_diff

def federatedModelWeightUpdate(randomClientIDs, client_ckpts, weights, save_path=None):
    '''
    To Do Updating
    '''
    print("Start average model weight update:")

    # Initialize the new model checkpoint
    fed_ckpt = client_ckpts[randomClientIDs[0]]

    # Move all parameters of the first client to CPU and cast to float
    fed_model = {k: v.cpu().float() for k, v in fed_ckpt['model'].items()}

    # Initialize new_ckpt to store the updated weights
    new_ckpt = {'model': {k: torch.zeros_like(v, device='cpu') for k, v in fed_model.items()}}

    # Iterate over all parameter names in the federated model
    for param_name, fed_param in fed_model.items():
        # Collect the weighted sum of parameters directly
        weighted_sum = torch.zeros_like(fed_param, device='cpu', dtype=torch.float32)

        # Aggregate weights from all clients
        for clientID in randomClientIDs:
            ckpt = client_ckpts[clientID]
            param = ckpt['model'][param_name].cpu().float()  # Ensure parameter is on CPU and cast to float
            weight = float(weights[clientID])  # Ensure weight is a float
            weighted_sum += param * weight
        new_ckpt['model'][param_name] = weighted_sum

    if save_path:
        misc.save_on_master(new_ckpt, save_path)
    print('model weight update is finished!')
    return new_ckpt

def client_resume_from_federated(model, args, vit_indexs, blocks_num, load_path='./checkpoint/federated/fed_checkpoint.pth'):
    # client model load the federated checkpoint according to the given vit block index
    if os.path.exists(load_path):
        fed_checkpoint = torch.load(load_path, map_location='cpu')
        new_checkpoint = {}
        allowed_blocks = ["blocks.%s." % i for i in range(blocks_num)]
        for k in fed_checkpoint['model']:
            if not "decoder" in k and "blocks" in k:
                for block_id in allowed_blocks:
                    if block_id in k:
                        new_checkpoint[k] = fed_checkpoint['model'][k]
                        break
            else:
                new_checkpoint[k] = fed_checkpoint['model'][k]
        for k in new_checkpoint:
            if not "decoder" in k and "blocks" in k:
                name_split = k.split(".")
                block_id = int(name_split[1])
                name_split[1] = str(vit_indexs[block_id])
                new_name = ".".join(name_split)
                new_checkpoint[k] = fed_checkpoint['model'][new_name]
        model.load_state_dict(new_checkpoint, strict=False)
        args.start_epoch = fed_checkpoint['epoch']
        print("Load checkpoint from: %s" % load_path)


def aggregate_from_clients(randomClientIDs, ckpts, 
                           data_len_ratios, blocks_nums, load_path='./checkpoint/federated/fed_checkpoint.pth'):

    """
    new way of aggregating the learned model gradients from clients
    Use case: 
        input: [id1, id2, id3]
        Structure of new global model:
            [prev base blocks] x n
            [id1_blocks_0] x n
            [id2_blocks_0] x n
            [id3_blocks_0] x n
            [after base blocks] x n
        return new global model
    """
    print("Start aggregating Vit blocks from clients")
    client_checkpoints = {}
    client_block_mappings = {}
    cps = {}

    for i in range(len(randomClientIDs)):
        clientID = randomClientIDs[i]
        ckpt = ckpts[clientID]
        block_num = blocks_nums[i]
        for block_id in block_num:
            if block_id in client_checkpoints:
                client_checkpoints[block_id].append(ckpt)
            else:
                client_checkpoints[block_id] = [ckpt]
            #client_checkpoints[block_id] = checkpoint
            client_block_mappings[block_id] = len(block_num)
        cps[clientID] = ckpt

    for k, v in client_checkpoints.items():
        if len(v) > 1:
            new_cp = copy.deepcopy(v[0])
            for kk in new_cp['model']:
                param_list = []
                for cp in v:
                    param_list.append(cp['model'][kk])
                mean_param = torch.mean(torch.stack(param_list), dim=0)
                new_cp['model'][kk] = mean_param
            client_checkpoints[k] = new_cp
        else:
            client_checkpoints[k] = v[0]

    assert os.path.exists(load_path), "fed_checkpoint.pth must exist for model aggregation"
    new_checkpoint = torch.load(load_path, map_location='cpu')

    contained_block_ids = set()
    for k in new_checkpoint['model']:
        if not "decoder" in k and "blocks" in k:
            name_split = k.split(".")
            block_id = name_split[1]
            contained_block_ids.add(block_id)
    assert len(contained_block_ids) == len(client_checkpoints.keys()), "the block length in the global model %s does not match with the sum of clients model length %s" % (len(contained_block_ids), len(client_checkpoints.keys()))

    for k in new_checkpoint['model']:
        if not "decoder" in k and "blocks" in k:
            name_split = k.split(".")
            block_id = int(name_split[1])
            client_checkpoint = client_checkpoints[block_id]
            respond_block_id = block_id % client_block_mappings[block_id]
            name_split[1] = str(respond_block_id)
            new_name = ".".join(name_split)
            new_checkpoint['model'][k] = client_checkpoint['model'][new_name]
        else:
            params = []
            for id, cp in cps.items():
                params.append(cp['model'][k] * data_len_ratios[id])
            sum_param = torch.sum(torch.stack(params), dim=0)
            new_checkpoint['model'][k] = sum_param

    misc.save_on_master(new_checkpoint, load_path)

    print('Block aggregation is finished!')

    return new_checkpoint

def aggregate_from_clients_both(randomClientIDs, ckpts, 
                           data_len_ratios, enc_blocks_nums, dec_blocks_nums, load_path='./checkpoint/federated/fed_checkpoint.pth'):
    
    client_enc_ckpts = {}
    client_dec_ckpts = {}
    client_enc_block_mappings = {}
    client_dec_block_mappings = {}
    cps = copy.deepcopy(ckpts)

    for i in range(len(randomClientIDs)):
        clientID = randomClientIDs[i]
        ckpt = ckpts[clientID]
        enc_block_num = enc_blocks_nums[i]
        dec_block_num = dec_blocks_nums[i]
        for block_id in enc_block_num:
            if block_id in client_enc_ckpts:
                client_enc_ckpts[block_id].append(ckpt)
            else:
                client_enc_ckpts[block_id] = [ckpt]
            client_enc_block_mappings[block_id] = len(enc_block_num)
        for block_id in dec_block_num:
            if block_id in client_dec_ckpts:
                client_dec_ckpts[block_id].append(ckpt)
            else:
                client_dec_ckpts[block_id] = [ckpt]
            client_dec_block_mappings[block_id] = len(dec_block_num)
    
    for k, v in client_enc_ckpts.items():
        if len(v) > 1:
            new_cp = copy.deepcopy(v[0])
            for kk in new_cp['model']:
                param_list = []
                for cp in v:
                    param_list.append(cp['model'][kk])
                mean_param = torch.mean(torch.stack(param_list), dim=0)
                new_cp['model'][kk] = mean_param
            client_enc_ckpts[k] = new_cp
        else:
            client_enc_ckpts[k] = v[0]

    for k, v in client_dec_ckpts.items():
        if len(v) > 1:
            new_cp = copy.deepcopy(v[0])
            for kk in new_cp['model']:
                param_list = []
                for cp in v:
                    param_list.append(cp['model'][kk])
                mean_param = torch.mean(torch.stack(param_list), dim=0)
                new_cp['model'][kk] = mean_param
            client_dec_ckpts[k] = new_cp
        else:
            client_dec_ckpts[k] = v[0]

    assert os.path.exists(load_path), "previous model weights must exist for model aggregation"
    new_checkpoint = torch.load(load_path, map_location='cpu')

    contained_enc_block_ids = set()
    contained_dec_block_ids = set()
    for k in new_checkpoint['model']:
        if "blocks" in k:
            name_split = k.split(".")
            block_id = name_split[1]
            if not "decoder" in k:
                contained_enc_block_ids.add(block_id)
            else:
                contained_dec_block_ids.add(block_id)
    assert len(contained_enc_block_ids) == len(client_enc_ckpts.keys()), "the encoder block length in the global model %s does not match with the sum of encoder block length in client models: %s" % (len(contained_enc_block_ids), len(client_enc_ckpts.keys()))
    assert len(contained_dec_block_ids) == len(client_dec_ckpts.keys()), "the decoder block length in the global model %s does not match with the sum of decoder block length in client models: %s" % (len(contained_dec_block_ids), len(client_dec_ckpts.keys()))

    for k in new_checkpoint['model']:
        if "blocks" in k:
            name_split = k.split(".")
            block_id = int(name_split[1])
            if not "decoder" in k:
                client_checkpoint = client_enc_ckpts[block_id]
                respond_block_id = block_id % client_enc_block_mappings[block_id]
                name_split[1] = str(respond_block_id)
                new_name = ".".join(name_split)
                new_checkpoint['model'][k] = client_checkpoint['model'][new_name]
            else:
                client_checkpoint = client_dec_ckpts[block_id]
                respond_block_id = block_id % client_dec_block_mappings[block_id]
                name_split[1] = str(respond_block_id)
                new_name = ".".join(name_split)
                new_checkpoint['model'][k] = client_checkpoint['model'][new_name]
                
        else:
            params = []
            for id, cp in cps.items():
                params.append(cp['model'][k] * data_len_ratios[id])
            sum_param = torch.sum(torch.stack(params), dim=0)
            new_checkpoint['model'][k] = sum_param

    misc.save_on_master(new_checkpoint, load_path)

    print('Block aggregation for both encoders and decoders is finished!')

    return new_checkpoint


def load_ckpt_and_freeze(client_ID, freeze_blocks, model, args):

    """
    model loads the global model weight from the checkpoint, and then
    freezes the specified blocks in encoder
    """
    
    load_path = './checkpoint/federated/fed_checkpoint.pth'
    if os.path.exists(load_path):
        fed_checkpoint = torch.load(load_path, map_location='cpu')
        model.load_state_dict(fed_checkpoint['model'])
        for _, param in model.named_parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if not "decoder" in name and "blocks" in name:
                name_split = name.split(".")
                block_id = int(name_split[1])
                if block_id in freeze_blocks[client_ID]:
                    param.requires_grad = False
        args.start_epoch = fed_checkpoint['epoch']
        print("Load checkpoint from: %s" % load_path)

def aggregate_sg_clients(randomClientIDs, client_ckpts, 
                         data_len_ratios, sg_mappings, record_round=0):
    
    """
    Aggregate the local client models which trained with stop gradient
    """

    save_path = './checkpoint/federated/fed_checkpoint.pth'
    new_fed_ckpt = copy.deepcopy(client_ckpts[randomClientIDs[0]])
    fed_model = copy.deepcopy(new_fed_ckpt['model'])
    for param_name in fed_model:
        if not "decoder" in param_name and "blocks" in param_name:
            name_split = param_name.split(".")
            block_id = int(name_split[1])
            respond_client = sg_mappings[block_id]
            client_model = client_ckpts[respond_client]['model']
            fed_model[param_name] = client_model[param_name]
        else:
            params = []
            for clientID in randomClientIDs:
                ckpt = client_ckpts[clientID]
                param = ckpt['model'][param_name] * data_len_ratios[clientID]
                params.append(param)
            fed_model[param_name] = torch.sum(torch.stack(params), dim=0)
    new_fed_ckpt['model'] = fed_model
    misc.save_on_master(new_fed_ckpt, save_path)
    if record_round:
        backup_path = './checkpoint/federated/fed_checkpoint_%srd.pth' % record_round
        misc.save_on_master(new_fed_ckpt, backup_path)
        print("Backup federated checkpoint at %s rounds" % record_round)

    print('federated model weight update is finished!')
    

def build_client_model_weight(model, args, vit_indexs, blocks_num):
    # client model load the federated checkpoint according to the given vit block index
    """
    vit_indexes: 
        example - {"encoder": [0, 1], "decoder": [0]}
    blocks_num:
        example - [2, 1]
    """
    load_path = './checkpoint/federated/fed_checkpoint.pth'
    if os.path.exists(load_path):
        fed_checkpoint = torch.load(load_path, map_location='cpu')
        new_checkpoint = {}
        allowed_enco_blks = ["blocks.%s." % i for i in range(blocks_num[0])]
        allowed_deco_blks = ["decoder_blocks.%s." % i for i in range(blocks_num[1])]
        for k in fed_checkpoint['model']:
            if not "decoder" in k and "blocks" in k:
                for block_id in allowed_enco_blks:
                    if block_id in k:
                        new_checkpoint[k] = fed_checkpoint['model'][k]
                        break
            elif "decoder_blocks" in k:
                for block_id in allowed_deco_blks:
                    if block_id in k:
                        new_checkpoint[k] = fed_checkpoint['model'][k]
                        break
            else:
                new_checkpoint[k] = fed_checkpoint['model'][k]
        for k in new_checkpoint:
            if not "decoder" in k and "blocks" in k:
                name_split = k.split(".")
                block_id = int(name_split[1])
                name_split[1] = str(vit_indexs["encoder"][block_id])
                new_name = ".".join(name_split)
                new_checkpoint[k] = fed_checkpoint['model'][new_name]
            elif "decoder_blocks" in k:
                name_split = k.split(".")
                block_id = int(name_split[1])
                name_split[1] = str(vit_indexs["decoder"][block_id])
                new_name = ".".join(name_split)
                new_checkpoint[k] = fed_checkpoint['model'][new_name]
        model.load_state_dict(new_checkpoint, strict=False)
        args.start_epoch = fed_checkpoint['epoch']
        print("Client load checkpoint from: %s" % load_path)
    


