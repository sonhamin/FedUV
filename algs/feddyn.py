import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
#https://github.com/mattiadutto/aml_federeted_learning/blob/main/fed-dyn.ipynb
ALPHA = 1e-3
def train_net_feddyn(net_id, net, global_net, train_dataloader, test_dataloader, epochs, 
                        all_previous_gradient, args, round, device, logger):
    
       
    global_net.to(device)
    net = net.to(device)

    c_previous_gradient = all_previous_gradient[net_id]

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    #train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    #logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    cnt = 0
    global_weight_collector = list(global_net.to(device).parameters())
    local_w = copy.deepcopy(net.state_dict())
    par_flat = torch.cat([p.reshape(-1) for p in global_net.parameters()])

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out, l1_local, l2_local = net(x)
            #_, pro2, _ = global_net(x)


            loss1 = criterion(out, target)
            
            cur_flat = torch.cat([p.reshape(-1) for p in net.parameters()])
            
            
            # Compute the linear penalty: prev_grad_flat · cur_flat
            linear_penalty = torch.sum(c_previous_gradient * cur_flat)
            # Compute the quadratic penalty: (alpha / 2) * || cur_flat - par_flat || ^ 2
            norm_penalty = (ALPHA / 2) * torch.linalg.norm(cur_flat - par_flat, 2) ** 2
            
            # Compute the total mini-batch loss
            loss = loss1 - linear_penalty + norm_penalty
            
            # Backward step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector) + 1e+4)
        epoch_loss1 = 0
        epoch_loss2 = 0
        logger.info('Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2))


        

    cur_flat = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    all_previous_gradient[net_id] -= ALPHA * (cur_flat - par_flat)
    
    #train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Training accuracy: %f' % train_acc)
    #logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Client: %d Training complete **' % (net_id))
    
    #q_w.put(net.state_dict())
    
    #time.sleep(3)
    return net.state_dict(), all_previous_gradient
    #return train_acc, test_acc