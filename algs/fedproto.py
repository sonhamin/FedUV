import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy 
PROTO_LD = 1
def train_net_fedproto(net_id, net, global_protos, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, round, device, logger):
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
    loss_mse  = nn.MSELoss().to(device)
    # criterion = nn.BCEWithLogitsLoss(reduction  = 'mean').to(device)

    # global_net.to(device)

    cnt = 0
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        agg_protos_label = {}
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device, non_blocking=True), target.type(torch.LongTensor).to(device, non_blocking=True)
            # x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)


            optimizer.zero_grad()
            protos, _, out, _, _ = net(x)
            loss1 = criterion(out, target)

            if len(global_protos) == 0:
                loss2 = 0*loss1
            else:
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in target:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)

            loss = loss1 + loss2 * PROTO_LD
            loss.backward()
            
            
            optimizer.step()

            for i in range(len(target)):
                if target[i].item() in agg_protos_label:
                    agg_protos_label[target[i].item()].append(protos[i,:])
                else:
                    agg_protos_label[target[i].item()] = [protos[i,:]]

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector) + 1e-8)
        logger.info('Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (net_id, epoch, epoch_loss, 0, 0))


    net.to('cpu')
    logger.info(' ** Client: %d Training complete **' % (net_id))
    
    return net.state_dict(), agg_protos_label