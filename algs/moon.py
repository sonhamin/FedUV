import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args, round, device, logger):
    net = net.cuda()
    global_net.cuda()

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

    for previous_net in previous_nets:
        previous_net.to(device)
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    #cos=kernel_CKA
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

            _, pro1, out, _, _ = net(x)
            _, pro2, _, _, _ = global_net(x)

            posi = cos(pro1, pro2)
            
            
            
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net = previous_net.cuda()
                _, pro3, _, _, _ = previous_net(x)
                nega = cos(pro1, pro3)
                                
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                
                #previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).to(device).long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()

                
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector)+1e-8)
        epoch_loss1 = sum(epoch_loss1_collector) / (len(epoch_loss1_collector)+1e-8)
        epoch_loss2 = sum(epoch_loss2_collector) / (len(epoch_loss2_collector)+1e-8)
        logger.info('Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    #train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Training accuracy: %f' % train_acc)
    #logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Client: %d Training complete **' % (net_id))
    
    return net.state_dict()
    #q_w.put(net.state_dict())
    
    #time.sleep(3)
    #return train_acc, test_acc
    
    
    
    
    
    
    
