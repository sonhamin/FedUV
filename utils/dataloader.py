import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import STL10, CIFAR10, ImageFolder, DatasetFolder, utils
import torchvision.transforms as transforms
import numpy as np
import torch


def record_net_data_stats(y_train, net_dataidx_map, logdir, logger):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def load_stl10_data(args, datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    stl10_train_ds = STL10(args.datadir, split='train', download=True, transform=transform)
    stl10_test_ds = STL10(args.datadir, split='test', download=True, transform=transform)

    X_train, y_train = stl10_train_ds.data, stl10_train_ds.labels
    X_test, y_test = stl10_test_ds.data, stl10_test_ds.labels

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.transpose(X_train, (0,2,3,1))
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.transpose(X_test, (0,2,3,1))

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(args, datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return (X_train, y_train, X_test, y_test)



def load_feature_shift(args):

    if args.dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch'] # pacs
        
    elif args.dataset == 'ham':

        domains = ['vidir_molemax', 'vidir_modern', 'rosendahl', 'vienna_dias'] # HAM
        
    elif args.dataset == 'office':
        domains = ['Art', 'Clipart', 'Product', 'Real'] # office

    all_train_ds = []
    for domain in domains:
        all_train_ds.append(dl_obj(f"{args.datadir}/{args.dataset}/train/{domain}", transform=transform_train))

    train_ds_global = torch.utils.data.ConcatDataset(all_train_ds)

    train_dl_global = data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, num_workers=8, shuffle=True, 
                             pin_memory=True, persistent_workers=True
                             )

    test_ds_global = dl_obj(f"{args.datadir}/{args.dataset}/test", transform=transform_test)
    test_dl = data.DataLoader(dataset=test_ds_global, batch_size=args.batch_size, num_workers=4, shuffle=False, 
                             pin_memory=True, persistent_workers=True
                             )
    
    return all_train_ds, train_ds_global, train_dl_global, test_ds_global, test_dl


def partition_data(args, dataset, datadir, logdir, partition, n_parties, beta=0.4, logger=None):
    
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(args, datadir)
    elif dataset == 'stl10':
        X_train, y_train, X_test, y_test = load_stl10_data(args, datadir)
        

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    ########### SEE IF WE NEED THIS LATER ##############
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir, logger)
    
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)



class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
        
        
class CustomDataset(Dataset):
    def __init__(self, data, targets, transforms=None):
        
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transforms = transforms

        print(self.data.shape)
        print(self.targets.shape)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        if self.transforms is not None:
            batch_x, batch_y = self.transforms(self.data[index]), self.targets[index]
        else:
            batch_x, batch_y = self.data[index], self.targets[index]

        return batch_x, batch_y
    

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        
        self.idxs = None
        if idxs is not None:
            self.idxs = list(idxs)
            
    def __len__(self):
        if self.idxs is not None:
            return len(self.idxs)
        else:
            return len(self.dataset)

    def __getitem__(self, item):
        if self.idxs is not None:
            image, label = self.dataset[self.idxs[item]]
        else:
            image, label = self.dataset[item]
        
        return image, label
    
    
from torch.autograd import Variable
import torch.nn.functional as F
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)     

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def get_dataloader(ds_name, datadir, train_bs, test_bs, X_train=None, y_train=None, X_test=None, y_test=None, dataidxs=None, noise_level=0):
    if ds_name in ('cifar10'):
        if ds_name == 'cifar10':
            
            dataset=CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.247, 0.243, 0.261])
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.ColorJitter(brightness=noise_level),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])


        train_ds = CustomDataset(X_train, y_train, transforms=transform_train)
        test_ds  = CustomDataset(X_test, y_test, transforms=transform_test)

        train_ds = DatasetSplit(train_ds, dataidxs)
        
   

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, num_workers=8, drop_last=True, shuffle=True, pin_memory=True, persistent_workers =True, worker_init_fn=set_worker_sharing_strategy)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, num_workers=8, shuffle=False,persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)

        
    elif ds_name == 'stl10':
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])        
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = CustomDataset(X_train, y_train, transforms=transform_train)
        test_ds  = CustomDataset(X_test, y_test, transforms=transform_test)

        train_ds = DatasetSplit(train_ds, dataidxs)
                

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, num_workers=8, drop_last=True, shuffle=True, pin_memory=True, persistent_workers =True, worker_init_fn=set_worker_sharing_strategy)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, num_workers=8, shuffle=False,persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)
        
        

    return train_dl, test_dl, train_ds, test_ds    
