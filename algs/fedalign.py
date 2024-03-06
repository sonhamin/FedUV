import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def transmitting_matrix(fm1, fm2, using_fc=True):
    if not using_fc:
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

    fsp = torch.bmm(fm1, fm2) / fm1.size(2)
    return fsp

def top_eigenvalue(K, n_power_iterations=10, dim=1, device=torch.device('cpu')):
    v = torch.ones(K.shape[0], K.shape[1], 1).to(device)
    for _ in range(n_power_iterations):
        m = torch.bmm(K, v)
        n = torch.norm(m, dim=1).unsqueeze(1)
        v = m / n

    top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
    return top_eigenvalue



def train_net_fedalign(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, round, device, logger):
    net = net.to(device)

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    #train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    #logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

#     for name, param in net.named_parameters():
#         print(name, param.requires_grad)      

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)

    # global_net.to(device)
    

    cnt = 0
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            x, target = x.to(device, non_blocking=True), target.type(torch.LongTensor).to(device, non_blocking=True)


            optimizer.zero_grad()
            t_feats0, t_feats1, out, _, _ = net(x)
            loss = criterion(out, target)

            loss.backward()
            
            #t_feats[-2] => t_feats0
            #t_feats[-1] => t_feats1         
            s_feats0, s_feats1 = net.reuse_feature(t_feats0.detach(), args.width_mult)


                            # Lipschitz loss
            TM_s = torch.bmm(transmitting_matrix(s_feats0, s_feats1, args.using_fc), 
                             transmitting_matrix(s_feats0, s_feats1, args.using_fc).transpose(2,1))
            TM_t = torch.bmm(transmitting_matrix(t_feats0.detach(), t_feats1.detach(), args.using_fc), 
                             transmitting_matrix(t_feats0.detach(), t_feats1.detach(), args.using_fc).transpose(2,1))
            
            loss_lip = F.mse_loss(top_eigenvalue(K=TM_s, device=device), top_eigenvalue(K=TM_t, device=device))
            
            loss_lip = args.mu*(loss.item()/loss_lip.item())*loss_lip
            
            loss_lip.backward()
            
#             if args.dataset == 'imbd':
#                 torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)            
            
            
            #print(f"loss_ce: {np.round(loss.item(), 3)}, loss_lip: {np.round(loss_lip.item(), 3)}")
            #Gradient Clipping
            if args.dataset == 'imbd':
                params = list(filter(lambda p: p.grad is not None, net.parameters()))
                for p in params:
                    p.grad.data.clamp_(-1e-1, 1e-1)                
            #~Gradient Clipping
            else:            
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector)+1e-8)
        logger.info('Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (net_id, epoch, epoch_loss, 0, 0))


    net.to('cpu')
    logger.info(' ** Client: %d Training complete **' % (net_id))
    
    return net.state_dict()