import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, round, device, logger):
    net = net.to(device)

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    #train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    #logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    cnt = 0
    global_weight_collector = list(global_net.to(device).parameters())
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            #x.requires_grad = True
            #target.requires_grad = False
            target = target.long()

            _, pro1, out, l1_local, l2_local = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg            
            
            loss.backward()

            
                
            optimizer.step()
            
       

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector)+1e-8)
        logger.info('Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (net_id, epoch, epoch_loss, 0, 0))


    net.to('cpu')
    logger.info(' ** Client: %d Training complete **' % (net_id))
    
    return net.state_dict()