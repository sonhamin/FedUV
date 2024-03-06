import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UVReg(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.args = args
        self.n_classes = n_classes
        self.soft = nn.Softmax(dim=1)

            
#         tester = torch.eye(self.args.batch_size)[:self.n_classes, :]
#         tester_soft = self.soft(tester)
#         self.batch_gamma = torch.sqrt(tester_soft.var(dim=0) + 0.0001)[0].item()       
        
        tester = torch.eye(self.n_classes)
#         while tester.shape[0] < self.args.batch_size:
#             tester = torch.cat((tester,tester), dim=0)
#         tester = tester[:self.args.batch_size]
        #tester_soft = self.soft(tester)
        
        self.batch_gamma = tester.std(dim=0).mean().item()#torch.sqrt(tester_soft.var(dim=0) + 0.0001)[0].item()        
        

    def forward(self, X, Y, pro1):

        if len(X.shape) > 2:
            X = torch.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
            Y = torch.reshape(Y, (Y.shape[0], np.prod(Y.shape[1:])))


        pdist_x = torch.pdist(pro1, p=2).pow(2)
        sigma_unif_x = torch.median(pdist_x[pdist_x != 0])

        unif_loss = pdist_x.mul(-1/sigma_unif_x).exp().mean()

        logsoft_out = self.soft(X)
        logsoft_out_std = logsoft_out.std(dim=0)
        std_loss = torch.mean(F.relu(self.batch_gamma - logsoft_out_std))

        loss = (
            self.args.std_coeff * std_loss
            + self.args.unif_coeff * unif_loss
        )

        return loss, np.array([0, np.round(std_loss.item(), 5),
                               0, 0,
                               np.round(unif_loss.item(), 10)])
    
    
def train_net_feduv(net_id, net, train_dataloader, test_dataloader, n_classes, epochs, lr, args_optimizer, args, round, device, logger):
    net = net.to(device)

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = UVReg(args, n_classes)   
    criterion_ce = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    cnt = 0
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss_mse_collector = []
        epoch_loss_std_collector = []
        epoch_loss_hsic_collector = []
        epoch_loss_unif_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()

            _, pro1, out, _, _ = net(x)
            
            loss_ce = criterion_ce(out, target)
                      
            target = torch.nn.functional.one_hot(target, num_classes=n_classes)
            target = target.float()
            
            
            loss, loss_logs = criterion(out, target, pro1)
            
            loss+=(loss_ce)


            loss.backward()

            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss_mse_collector.append(loss_ce.item()) #epoch_loss_mse_collector.append(loss_logs[0])
            epoch_loss_std_collector.append(loss_logs[1])
            epoch_loss_hsic_collector.append(loss_logs[3])
            epoch_loss_unif_collector.append(loss_logs[4])

        epoch_loss = np.round(sum(epoch_loss_collector) / (len(epoch_loss_collector) + 1e-8), 5)
        epoch_loss_mse = np.round(sum(epoch_loss_mse_collector) / (len(epoch_loss_mse_collector) + 1e-8), 5)
        epoch_loss_std = np.round(sum(epoch_loss_std_collector) / (len(epoch_loss_std_collector) + 1e-8), 5)
        epoch_loss_hsic= np.round(sum(epoch_loss_hsic_collector) / (len(epoch_loss_hsic_collector) + 1e-8), 5)
        epoch_loss_unif= np.round(sum(epoch_loss_unif_collector) / (len(epoch_loss_unif_collector) + 1e-8), 10)
        logger.info(f"Client: {net_id} Epoch: {epoch} Loss: {epoch_loss} mse: {epoch_loss_mse} std: {epoch_loss_std} hsic: {epoch_loss_hsic} unif: {epoch_loss_unif}")
        #print(f"Client: {net_id} Epoch: {epoch} Loss: {epoch_loss} mse: {epoch_loss_mse} std: {epoch_loss_std} hsic: {epoch_loss_hsic} unif: {epoch_loss_unif}")


    net.to('cpu')
    logger.info(' ** Client: %d Training complete **' % (net_id))
    return net.state_dict()
    


