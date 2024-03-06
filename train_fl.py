import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random

import argparse

from torch.autograd import Variable

import time
import torch.nn.functional as F
import numpy as np
import copy

from utils.dataloader import partition_data
from utils.dataloader import DatasetSplit, get_dataloader

from nets.models import SimpleCNN
from nets.models import ResNet_18
from nets.models import ResNet_50

from algs.fedavg import train_net_fedavg
from algs.fedprox import train_net_fedprox
from algs.moon import train_net_moon
from algs.feduv import train_net_feduv

from utils.calculate_acc import compute_accuracy


def init_nets(n_parties, args, device, n_classes):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        
        if args.model == 'resnet-50':
            net = ResNet_50(args, n_classes)
        elif args.model == 'resnet-18':
            net = ResNet_18(args, n_classes)
        elif args.model == 'simple-cnn':
            net = SimpleCNN(args.out_dim, n_classes, args.simp_width)
        nets[net_i] = net


    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
        
    return nets, model_meta_data, layer_type



def main(args):

    #0. Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)    
    
    #1. check device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""    

    if args.device == 'cuda' and device.type == 'cpu':
        print("GPU not detected, defaulting to CPU")

    print("Device: ", device.type)

    if device.type == 'cuda':
        args.multiprocessing=0
        
        
    print(f"Algorithm: {args.alg}")
        
        
    #2. Init logs
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'

    from importlib import reload
    reload(logging)


    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    

    #3. Get data
    
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
            
            
    if args.dataset == 'stl10'or args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            args,
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.alpha, 
           logger=logger
        )
        
        n_classes = len(np.unique(y_train))
        train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   args.batch_size,
                                                                                   X_train, y_train,
                                                                                   X_test, y_test)

        
        train_dl_local_list, test_dl_local_list = [],[]
        for i in range(args.n_parties):

            dataidxs = net_dataidx_map[i]
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, 
                                                                 args.datadir, 
                                                                 args.batch_size, 
                                                                 args.batch_size, 
                                                                 X_train, y_train,
                                                                 X_test, y_test,
                                                                 dataidxs)

            train_dl_local_list.append(train_dl_local)
            test_dl_local_list.append(test_dl_local)


        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, 
                                                               args.datadir, 
                                                               args.batch_size, 
                                                               args.batch_size,
                                                               X_train, y_train,
                                                               X_test, y_test)    
        
    else:
        all_train_ds, train_ds_global, train_dl_global, test_ds_global, test_dl = load_feature_shift(args)
        
        n_classes = len(np.unique(test_ds_global.samples[:,1]))

        all_train_ds = []
        for domain in args.domains:
            all_train_ds.append(dl_obj(f"{args.datadir}/{args.dataset}/train/{domain}", transform=transform_train))

        train_ds_global = torch.utils.data.ConcatDataset(all_train_ds)

        train_dl_global = data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, num_workers=8, shuffle=True, 
                                 pin_memory=True, persistent_workers=True
                                 )


        test_ds_global = dl_obj(f"{args.datadir}/{args.dataset}/test", transform=transform_test)
        test_dl = data.DataLoader(dataset=test_ds_global, batch_size=args.batch_size, num_workers=4, shuffle=False, 
                                 pin_memory=True, persistent_workers=True
                                 )
        
        
    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    print("len test_dl_global:", data_size)
    print(f"n_classes: {n_classes}")
        
        
        
        
        
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args, device, n_classes)

    global_models, global_model_meta_data, global_layer_type = init_nets(1, args, device, n_classes)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    if args.alg == 'moon':
        old_nets_pool = []

        if args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
        
        

    


    n_epoch = args.epochs
    global_optimizer = optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.reg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(global_optimizer, n_comm_rounds)

    for round in range(n_comm_rounds):
        print("\n\n************************************")
        print("round: ", round)
        cur_time = time.time()
        logger.info("in comm round:" + str(round))
  



        cur_lr = scheduler.get_last_lr()[0]    
        print(f"Current LR: {cur_lr}")

        party_list_this_round = party_list_rounds[round]

        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False
        global_w = global_model.state_dict()


        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)



        if args.alg == 'Freeze':
            print("Freezing Weights")
            for net in nets_this_round.values():
                for param in net.fc3.parameters():
                    param.requires_grad = False                


        avg_acc = 0.0
        acc_list = []
        if global_model:
            global_model.to(device)



        local_weights = []


        procs=[]
        w_locals = []
        c_locals = []   #Only used for scaffold
        c_deltas = []   #Only used for scaffold
        for net_id, net in nets_this_round.items():

            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            train_dl_local, test_dl_local = train_dl_local_list[net_id], test_dl_local_list[net_id]

            if args.alg == 'moon':

                prev_models=[]

                for i in range(len(old_nets_pool)):
                    prev_models.append(old_nets_pool[i][net_id])


                local = train_net_moon(net_id, net, global_model, prev_models, 
                                              train_dl_local, test_dl, n_epoch, cur_lr,
                                              args.optimizer, args.mu, args.temperature, 
                                              args, round, device, logger)
                w_locals.append(local)            

            elif args.alg == 'fedprox':

                single_local = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, cur_lr,
                                                              args.optimizer, args, round, 
                                                             device, logger)

                w_locals.append(single_local)



            elif args.alg == 'fedavg' or args.alg == 'freeze':


                single_local = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, cur_lr,
                                                              args.optimizer, args, round, 
                                                            device, logger)

                w_locals.append(single_local)

            elif args.alg == 'feduv':
                single_local = train_net_feduv(net_id, net, train_dl_local, test_dl, n_classes, n_epoch, cur_lr, 
                                  args.optimizer, args, round, device, logger)

                w_locals.append(single_local)



        for count, net in enumerate(nets_this_round.values()):
            net.load_state_dict(w_locals[count])               


        avg_acc /= args.n_parties
        if args.alg == 'local_training':
            logger.info("avg test acc %f" % avg_acc)
            logger.info("std acc %f" % np.std(acc_list))
        if global_model:
            global_model.to('cpu')




        print("TIME: ", time.time()-cur_time)
        scheduler.step()


        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


        layer_exclude_list = []
        if args.alg=='freeze':
            layer_exclude_list.append("fc3")
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    if not any(exclude_key in key for exclude_key in layer_exclude_list):
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    if not any(exclude_key in key for exclude_key in layer_exclude_list):
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


        global_model.load_state_dict(global_w)

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl))
        global_model.to(device)
        
        train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
        test_acc, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)
        global_model.to('cpu')
        logger.info('>> Global Model Train accuracy: %f' % train_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Train loss: %f' % train_loss)

        print('>> Global Model Train accuracy: %f' % train_acc)
        print('>> Global Model Test accuracy: %f' % test_acc)
        print('>> Global Model Train loss: %f' % train_loss)


        if args.alg == 'moon':
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets



        print("TIME: ", time.time()-cur_time)
        print("************************************\n\n")
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication rounds')
    parser.add_argument('--sample_fraction', type=float, default=1.0, 
                        help='how many clients are sampled in each round')    
    parser.add_argument('--alpha', type=float, default=0.01, 
                        help='The parameter for the dirichlet distribution for data partitioning')
    
    parser.add_argument('--dataset', type=str, default='stl10', help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="../data/", help="Data directory")
    parser.add_argument('--partition', type=str, required=False, default='noniid', help='the data partitioning strategy')
    
    parser.add_argument('--alg', type=str, default='feduv',
                        help='federated learning framework: fedavg/fedprox/moon/freeze/feduv')    
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')   
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for FedProx or MOON')    
    
    parser.add_argument('--std_coeff', type=float, default=2.5, help='the lambda parameter for FedUV')
    parser.add_argument('--unif_coeff', type=float, default=0.5, help='the mu parameter for FedUV')

    
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='input batch size for training')
    
    parser.add_argument('--load_first_net', type=int, default=1, 
                        help='whether load the first net as old net or not')
    parser.add_argument('--pool_option', type=str, default='FIFO', 
                        help='whether load the first net as old net or not')    
    
    
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')    
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    
    parser.add_argument('--simp_width', type=int, default=1, help='multiplier for CNN channel width (only for simple-cnn)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='the temperature parameter for contrastive loss')
    
    parser.add_argument('--logdir', type=str, required=False, default="./", help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')


    parser.add_argument('--seed', type=int, default=42, help='The seed number')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program (cuda/cpu)')
    
    
    args = parser.parse_args()    
    main(args)
    
        
        
        
        
        
        
        
        