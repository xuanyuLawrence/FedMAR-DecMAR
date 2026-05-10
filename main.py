import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import argparse
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset import *
from train_server import finetune_VIT, finetune_CNN
from train_client import pretrain_MAE, pretrain_MAE_CNN, pretrain_contra_CNN, pretrain_contra_ViT
from federated_learning import federatedModelWeightUpdate, aggregate_from_clients, aggregate_from_clients_both
from debug import ckpt_diff
import util.misc as misc
import copy

import model_ViT
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import PIL
import prepare_dataset as pdata
import baseline_models as bm
import decentralized as decen
import gc
import ammd
import model_mae
import train_client 


def get_argparser():
    parser = argparse.ArgumentParser(description="Run non-iid study")
    parser.add_argument('-rd', '--rounds', default=100, type=int,
                        help='The number of rounds for federated learning')
    parser.add_argument('-tc', '--totalNum_clients', default=100, type=int,
                        help='The total number of client agents used in demo')
    parser.add_argument('-sc', '--num_of_Clients', default=5, type=int,
                        help='The number of client agents sampled in each round') 
    parser.add_argument('-le', '--num_of_local_epochs', default=2, type=int,
                        help='The number of epochs in local training') 
    parser.add_argument('-ced', '--client_encoder_depth', default=9, type=int,
                        help='The number of transformer blocks in client encoder')
    parser.add_argument('-cdd', '--client_decoder_depth', default=1, type=int,
                        help='The number of transformer blocks in client decoder')     
    parser.add_argument('-bed', '--backbone_encoder_depth', default=9, type=int,
                        help='The number of transformer blocks in backbone encoder')
    parser.add_argument('-bdd', '--backbone_decoder_depth', default=1, type=int,
                        help='The number of transformer blocks in backbone decoder') 
    parser.add_argument('-si', '--save_interval', default=10, type=int,
                        help='The number of epochs that saves the model')  
    parser.add_argument('-ri', '--record_interval', default=10, type=int, 
                        help='The round interval of saving checkpoint') 

    parser.add_argument('-sp', '--save_path', default='./ckpt_fedmar_mean/', type=str, help='checkpoint save path')
    parser.add_argument('-gp', '--graph_path', default='./graph/network_G_10avg.npy', type=str, help='network graph path') 

    parser.add_argument('-d', '--dataset', default=4, type=int, 
                        help='The id of dataset to use: 0 - CIFAR10; 1 - CIFAR100; 2 - Food101; 3 - ImageNet; 4 - Mini-ImageNet; 5 - Road-Sign, 6 - Mini-INAT;')
    parser.add_argument('-ra', '--ratio', default=1, type=float,
                        help='The ratio of labelled images')
    parser.add_argument('-samp', '--sampling', default="dir", 
                        help='The way of samping, iid, dir, shard or feature_skew')
    parser.add_argument('--alpha', default=1e-1, type=float, 
                        help='The required parameter for dir sampling, which decides the statistical heterogenity')
    parser.add_argument('-bl', '--baseline', default=0, type=int, 
                        help='The baselines: 0 - MAE+ViT; 1 - Simsiam+CNN, 2 - MAE+CNN, 3 - Siamsiam+ViT')
    parser.add_argument('-so', '--scenario', default=0, type=int, 
                        help='The baselines: 0 - Federated; 1 - Decentralized')
    parser.add_argument('-mc', '--avg_connectivity', default=10, type=int, 
                        help='The average connectivity of the decentralized network') 
    parser.add_argument('-gmax', '--gamma_max', default=1e-1, type=float, 
                        help='The upper bound of ratio for combining the alignment loss (i.e., loss = mae_loss + gamma * alignment_loss)')   
    parser.add_argument('-gmin', '--gamma_min', default=1e-3, type=float, 
                        help='The upper bound of ratio for combining the alignment loss (i.e., loss = mae_loss + gamma * alignment_loss)')    
    parser.add_argument('-mask', '--use_align_mask', default=0, type=int, 
                        help='If using aligned mask to control local to global distance computation: 0 - False; 1 - True') 
    parser.add_argument('-am', '--agg_mode', default=0, type=int, 
                        help='The model aggregation mode on server: 0 - aggregate by data volume; 1 - aggregrate by data volume and feature distance')                               
    
    parser.add_argument('-p', '--phase', default="pretrain",  
                        help='specify the codes to: pretrain, finetune')
    parser.add_argument('-ftd', '--ft_depth', default=5, type=int,  
                        help='The depth of model in finetuning')
    parser.add_argument('--seed', default=0, type=int,  
                        help='The seed used for reproducibility')
    return parser
    
def main(args):
    print ("Run Self-supervised learning non-iid study")

    since1 = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(args.save_path):
        os.makedirs('%sclient/' % (args.save_path), exist_ok=True)
        os.makedirs('%sserver/' % (args.save_path), exist_ok=True)
        os.makedirs('%sfinetune/' % (args.save_path), exist_ok=True)


    dataset_names = ["cifar10", "cifar100", "food101", "imagenet", "mini", "imagenet32", "road_sign", "inat"]
    dataset = dataset_names[args.dataset]

    mean, std = find_dataset_mean_std(dataset)
    print("Chosen dataset is %s" % dataset)

    if args.baseline == 0 or args.baseline == 2:
        train_transform, val_transform = pdata.init_pretrain_transform(mean, std)
    else:
        train_transform, val_transform = bm.create_transform(224, mean, std)

    train_dataset, _, num_classes = init_dataset(train_transform, val_transform, dataset=dataset)

    clientIDlist = [f'clientID_{i+1}' for i in range(args.totalNum_clients)]
    for cid in clientIDlist:
        os.makedirs('%sclient/%s/' % (args.save_path, cid), exist_ok=True)

    
    if args.phase == "pretrain":
        unsuper_train_idxs = np.arange(len(train_dataset))
        np.random.shuffle(unsuper_train_idxs)
        if args.sampling == "iid":
            args.sampling = "dir"
            args.alpha = 1e5
        if args.sampling == "feature_skew":
            unsupervised_datasets = get_unsuper_datasets(train_dataset, unsuper_train_idxs, clientIDlist, sampling=args.sampling, alpha=args.alpha, dataset=dataset, num_classes=num_classes, pre_transforms=(train_transform, dataset))
        else:
            unsupervised_datasets = get_unsuper_datasets(train_dataset, unsuper_train_idxs, clientIDlist, sampling=args.sampling, alpha=args.alpha, dataset=dataset, num_classes=num_classes)
        print("Choose %s Sampling!" % args.sampling)
    else:
        assert args.ratio > 0. and args.ratio <= 1.0, "divide ratio should be between 0 and 1"
        if args.ratio < 1.0:
            super_train_idxs, _ = pdata.divide_dataset(train_dataset, args.ratio, dataset)
        else:
            super_train_idxs = np.arange(len(train_dataset)) 
        supervised_dataset = pdata.get_super_dataset(train_dataset, super_train_idxs)
        unsupervised_datasets = None
        print("Dataset under ratio %s is created!" % args.ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()


    resent_depth = 18
    r_eps = args.num_of_local_epochs

    print("%s" % args.phase)

    if args.phase == "pretrain":

        transformer_depths = [args.client_encoder_depth, args.client_decoder_depth]

        if args.scenario == 0:
            print("Choose federated scenario!")

            server_load_ckpt_path = '%sserver/fed_checkpoint.pth' % (args.save_path)
            if args.baseline == 0:
                print("The chosen baseline is ")
                _ = pretrain_MAE(args, None, None, transformer_depths, r_eps=1, load_path=None, save_path=server_load_ckpt_path)
            elif args.baseline == 1:
                _ = pretrain_contra_CNN(args, None, None, resent_depth, r_eps=1, load_path=None, save_path=server_load_ckpt_path)
            elif args.baseline == 2:
                _ = pretrain_MAE_CNN(args, None, None, resent_depth, r_eps=1, load_path=None, save_path=server_load_ckpt_path)
            else:
                _ = pretrain_contra_ViT(args, None, None, transformer_depths, r_eps=1, load_path=None, save_path=server_load_ckpt_path)
            
            global_latent = None
            local_latents = {}

            # client_gamma = {cid: {'gamma': args.gamma_max, 'H_prev': None} for cid in clientIDlist}
            client_gamma = {cid: {'gamma': args.gamma_max, 'count': 0} for cid in clientIDlist}
            client_local_latent = {cid: {'latent': None} for cid in clientIDlist}

            print("Federated Learning will run for %s rounds" % args.rounds)
            for i in range(args.rounds):
                print("#"*30)
                print("Federated Learning Round %s/%s" % (i+1, args.rounds))
                print("#"*30)
                randomClientIDs = random.sample(clientIDlist, args.num_of_Clients)
                # randomClientIDs = clientIDlist[:4] + clientIDlist[-1]

                data_len_ratios = {}
                data_len_total = 0
                for clientID in randomClientIDs:
                    data_len_ratios[clientID] = len(unsupervised_datasets[clientID])
                    data_len_total += len(unsupervised_datasets[clientID])
                for clientID in data_len_ratios:
                    data_len_ratios[clientID] = data_len_ratios[clientID] / data_len_total

                cur_rd = i + 1


                print("^"*20)
                print("Client Training Starts")
                print("^"*20)
                ckpts = {}
                # generate align mask
                
                train_args = train_client.get_args_parser()
                train_args = train_args.parse_args()
                local_model = model_mae.__dict__[train_args.model](depth=transformer_depths[0], decoder_depth=transformer_depths[1], norm_pix_loss=train_args.norm_pix_loss)
                if args.use_align_mask != 0:
                    N = train_args.batch_size
                    L = (224 // local_model.patch_size) ** 2
                    align_mask_noise = torch.rand(N, L)
                else:
                    align_mask_noise = None

                for j in range(len(randomClientIDs)):
                    clientID = randomClientIDs[j]
                    client_save_path = '%sclient/%s/local_model.pth' % (args.save_path, clientID)
                    if args.baseline == 0:
                        ckpt, local_latent = pretrain_MAE(
                            args, clientID, unsupervised_datasets[clientID], transformer_depths, r_eps=r_eps, 
                            global_latent=global_latent, gamma=client_gamma, mask_noise=align_mask_noise, 
                            load_path=server_load_ckpt_path, save_path=client_save_path
                        )
                        if isinstance(local_latent, torch.Tensor):
                            local_latents[clientID] = local_latent
                    elif args.baseline == 1:
                        ckpt = pretrain_contra_CNN(args, clientID, unsupervised_datasets[clientID], resent_depth, r_eps=r_eps,  load_path=server_load_ckpt_path, save_path=client_save_path)
                    elif args.baseline == 2:
                        ckpt = pretrain_MAE_CNN(args, clientID, unsupervised_datasets[clientID], resent_depth, r_eps=r_eps,  load_path=server_load_ckpt_path, save_path=client_save_path)
                    else:
                        ckpt = pretrain_contra_ViT(args, clientID, unsupervised_datasets[clientID], transformer_depths, r_eps=r_eps,  load_path=server_load_ckpt_path, save_path=client_save_path)
                    ckpts[clientID] = ckpt

                    # update gamma per round at server
                    # if args.baseline == 0:
                        # local_model.load_state_dict(ckpt['model'], strict=False)
                        # global_model = model_mae.__dict__[train_args.model](depth=transformer_depths[0], decoder_depth=transformer_depths[1], norm_pix_loss=train_args.norm_pix_loss)
                        # if server_load_ckpt_path and os.path.exists(server_load_ckpt_path):
                        #     ckpt_fed = torch.load(server_load_ckpt_path, map_location='cpu')
                        #     global_model.load_state_dict(ckpt_fed['model'], strict=False)
                        # ammd.update_client_gamma(clientID, client_gamma, local_model, global_model, args.gamma_max, args.gamma_min)
                
                # update gamma per round at server
                if args.baseline == 0:
                    select_cap = (args.num_of_Clients * args.rounds) // args.totalNum_clients
                    ammd.update_gamma_by_cosine(client_gamma, randomClientIDs, args.gamma_max, args.gamma_min, select_cap)
                        

                print("^"*20)
                print("Federated Average Starts")
                print("^"*20)
                
                if len(local_latents.keys()) > 0 and local_latents[clientID] is not None:
                    latent_list = [latent for cid, latent in local_latents.items()]
                    global_latent = sum(latent_list) / len(latent_list)
                    # global_latent = ammd.geometric_median_tensor(latent_list)
                    del latent_list
                if args.agg_mode == 0:
                    fed_ckpt = federatedModelWeightUpdate(randomClientIDs, ckpts, data_len_ratios, save_path=server_load_ckpt_path)
                else:
                    # new aggregation with feature distance
                    g_vec = ammd.pool_latent(global_latent)  # [D] float32 CPU

                    client_distance = {}  # {cid: d_k}
                    for cid, loc_lat in local_latents.items(): 
                        u = ammd.pool_latent(loc_lat)              
                        d = ammd.cosine_distance_vec(u, g_vec)      
                        client_distance[cid] = float(d)

                    agg_weights = ammd.mix_weights(randomClientIDs, data_len_ratios=data_len_ratios, distance_dict=client_distance)

                    # client_distance = {}  # {cid: d_k}
                    # for cid, loc_lat in local_latents.items():       
                    #     client_distance[cid] = float(ammd.mse_distance_vec(loc_lat, global_latent))
                    # agg_weights = ammd.mix_weights(randomClientIDs, data_len_ratios=data_len_ratios, distance_dict=client_distance)
                    fed_ckpt = federatedModelWeightUpdate(randomClientIDs, ckpts, agg_weights, save_path=server_load_ckpt_path)
                local_latents = {}
                

                if cur_rd % args.record_interval == 0:
                    progress_backup_path = '%sserver/fed_checkpoint_%srd.pth' % (args.save_path, cur_rd)
                    misc.save_on_master(fed_ckpt, progress_backup_path)
                
                if cur_rd == args.rounds:
                    final_model_path = '%sserver/fed_checkpoint_final.pth' % (args.save_path)
                    misc.save_on_master(fed_ckpt, final_model_path)
                
                del fed_ckpt
                torch.cuda.empty_cache()


        else:
            print("Choose decentralized scenario!")

            uniform_neighbors = True

            # Create decentralzied network 
            print("^"*20)
            print("Create decentralzied network!")
            print("^"*20)
            if os.path.exists(args.graph_path):
                adj_matrix = decen.load_adj_matrix(args.graph_path)
                is_valid = decen.verify_adj_matrix_avg(adj_matrix, len(clientIDlist), args.avg_connectivity)
                if not is_valid:
                    adj_matrix = decen.generate_adj_matrix_avg(len(clientIDlist), args.avg_connectivity, uniform_neighbors)
                    decen.save_adj_matrix(adj_matrix, args.graph_path)
            else:
                adj_matrix = decen.generate_adj_matrix_avg(len(clientIDlist), args.avg_connectivity, uniform_neighbors)
                decen.save_adj_matrix(adj_matrix, args.graph_path)

            # Init local model on all clients
            if args.baseline == 0:
                print("The chosen baseline is MAE+ViT!")
                ckpt, _ = pretrain_MAE(args, None, None, transformer_depths, r_eps=1, load_path=None)
            elif args.baseline == 1:
                print("The chosen baseline is Simsiam+CNN!")
                ckpt = pretrain_contra_CNN(args, None, None, resent_depth, r_eps=1, load_path=None)
            elif args.baseline == 2:
                print("The chosen baseline is MAE+CNN!")
                ckpt = pretrain_MAE_CNN(args, None, None, resent_depth, r_eps=1, load_path=None)
            else:
                print("The chosen baseline is Simsiam+ViT!")
                ckpt = pretrain_contra_ViT(args, None, None, transformer_depths, r_eps=1, load_path=None)

            local_latents = {}
            for cid in clientIDlist:
                init_path = '%sclient/%s/local_model.pth' % (args.save_path, cid)
                misc.save_on_master(ckpt, init_path)
                local_latents[cid] = None

            print("Decentralized Learning will run for %s rounds" % args.rounds)
            for i in range(args.rounds):
                cur_rd = i + 1

                print("#"*30)
                print("Decentralized Learning Round %s/%s" % (cur_rd, args.rounds))
                print("#"*30)

                print("^"*20)
                print("Client Training Starts")
                print("^"*20)
                ckpts = {}
                for j in range(len(clientIDlist)):
                    clientID = clientIDlist[j]
                    client_load_path = '%sclient/%s/local_model.pth' % (args.save_path, clientID)
                    if args.baseline == 0:
                        ckpt, local_latent = pretrain_MAE(args, clientID, unsupervised_datasets[clientID], transformer_depths, r_eps=r_eps, global_latent=local_latents[clientID],  load_path=client_load_path, save_path=client_load_path)
                        if args.gamma > 0:
                            if isinstance(local_latent, torch.Tensor):
                                local_latents[clientID] = local_latent
                        else:
                            del local_latent
                            torch.cuda.empty_cache()
                    elif args.baseline == 1:
                        ckpt = pretrain_contra_CNN(args, clientID, unsupervised_datasets[clientID], resent_depth, r_eps=r_eps,  load_path=client_load_path, save_path=client_load_path)
                    elif args.baseline == 2:
                        ckpt = pretrain_MAE_CNN(args, clientID, unsupervised_datasets[clientID], resent_depth, r_eps=r_eps,  load_path=client_load_path, save_path=client_load_path)
                    else:
                        ckpt = pretrain_contra_ViT(args, clientID, unsupervised_datasets[clientID], transformer_depths, r_eps=r_eps,  load_path=client_load_path, save_path=client_load_path)
                    ckpts[clientID] = ckpt
                

                print("^"*20)
                print("Model Aggregation between neighbors starts")
                print("^"*20)
                new_ckpts = {}
                new_local_latents = {}
                for cid in clientIDlist:
                    new_local_latents[cid] = None
                if args.avg_connectivity < len(clientIDlist):
                    for j in range(len(clientIDlist)):
                        clientID = clientIDlist[j]
                        neighbors_list = [j] + decen.get_neighbors(adj_matrix, j)
                        neighbors_ID_list = [clientIDlist[nodeid] for nodeid in neighbors_list]
                        neighbor_latents = []

                        data_len_ratios = {}
                        data_len_total = 0
                        for nid in neighbors_ID_list:
                            data_len_ratios[nid] = len(unsupervised_datasets[nid])
                            data_len_total += len(unsupervised_datasets[nid])
                            neighbor_latents.append(local_latents[nid])
                        for nid in data_len_ratios:
                            data_len_ratios[nid] = data_len_ratios[nid] / data_len_total

                        if len(neighbor_latents) > 0 and neighbor_latents[0] is not None:
                            global_latent = sum(neighbor_latents) / len(neighbor_latents)
                            new_local_latents[clientID] = global_latent

                        temp_save_path = '%sserver/%s_agg_model.pth' % (args.save_path, clientID)
                        fed_ckpt = federatedModelWeightUpdate(neighbors_ID_list, ckpts, data_len_ratios, save_path=temp_save_path)
                        del fed_ckpt
                        torch.cuda.empty_cache()
                else:
                    neighbor_latents = []
                    global_latent = None

                    data_len_ratios = {}
                    data_len_total = 0
                    for cid in clientIDlist:
                        data_len_ratios[cid] = len(unsupervised_datasets[cid])
                        data_len_total += len(unsupervised_datasets[cid])
                        neighbor_latents.append(local_latents[cid])
                    for cid in data_len_ratios:
                        data_len_ratios[cid] = data_len_ratios[cid] / data_len_total
                    new_ckpt = federatedModelWeightUpdate(clientIDlist, ckpts, data_len_ratios)
                    if len(neighbor_latents) > 0 and neighbor_latents[0] is not None:
                        global_latent = sum(neighbor_latents) / len(neighbor_latents)
                    for cid in clientIDlist:
                        new_ckpts[cid] = new_ckpt
                        new_local_latents[cid] = global_latent

                local_latents = copy.deepcopy(new_local_latents)
                del new_local_latents
                torch.cuda.empty_cache()
                gc.collect()

                # Update local model after aggregation
                del ckpts
                torch.cuda.empty_cache()
                print("^"*20)
                print("Update local models!")
                print("^"*20)

                for cid in clientIDlist:
                    temp_save_path = '%sserver/%s_agg_model.pth' % (args.save_path, cid)
                    client_save_path = '%sclient/%s/local_model.pth' % (args.save_path, cid)
                    progress_backup_path = '%sclient/%s/local_model_%srd.pth' % (args.save_path, cid, cur_rd)
                    
                    if args.avg_connectivity < len(clientIDlist):
                        new_local_ckpt = torch.load(temp_save_path, map_location='cpu')
                        misc.save_on_master(new_local_ckpt, client_save_path)
                        if cur_rd % args.record_interval == 0:
                            misc.save_on_master(new_local_ckpt, progress_backup_path)
                    else:
                        misc.save_on_master(new_ckpts[cid], client_save_path)
                        if cur_rd % args.record_interval == 0:
                            misc.save_on_master(new_ckpts[cid], progress_backup_path)

                del new_ckpts
                torch.cuda.empty_cache()
                gc.collect()

                # Output the global model if the pretraining finishes
                if cur_rd == args.rounds:
                    print("^"*20)
                    print("Pretraining Finishes! Generate Global Model!")
                    print("^"*20)
                    data_len_ratios = {}
                    data_len_total = 0
                    for clientID in clientIDlist:
                        data_len_ratios[clientID] = len(unsupervised_datasets[clientID])
                        data_len_total += len(unsupervised_datasets[clientID])
                    for clientID in data_len_ratios:
                        data_len_ratios[clientID] = data_len_ratios[clientID] / data_len_total

                    global_model_path = '%sserver/ckpt_final.pth' % (args.save_path)

                    new_ckpts = {}

                    for clientID in clientIDlist:
                        client_save_path = '%sclient/%s/local_model.pth' % (args.save_path, clientID)
                        local_model = torch.load(client_save_path, map_location='cpu')
                        new_ckpts[clientID] = local_model

                    federatedModelWeightUpdate(clientIDlist, new_ckpts, data_len_ratios, save_path=global_model_path)
                
    else:

        transformer_depths = [args.backbone_encoder_depth, args.backbone_decoder_depth]

        mean, std = find_dataset_mean_std(dataset)

        if args.scenario == 0:
            final_model_path = '%sserver/fed_checkpoint_final.pth' % (args.save_path)
        else:
            final_model_path = '%sserver/ckpt_final.pth' % (args.save_path)

        if args.baseline == 0 or args.baseline == 3:
            finetune_VIT(args, super_train_idxs, mean, std, dataset, args.ratio, transformer_depths[0], load_path=final_model_path)
        else:
            finetune_CNN(args, super_train_idxs, mean, std, dataset, args.ratio, resent_depth, load_path=final_model_path)

    del device

    time_elapsed1 = time.time() - since1
    print('Federated Learning complete in {:.0f}m {:.0f}s'.format(time_elapsed1 // 60, time_elapsed1 % 60))



if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)