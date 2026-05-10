import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import random
import PIL
from sklearn.model_selection import train_test_split
from timm.data import create_transform
from util.crop import RandomResizedCrop
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from imagenetLoad import ImageNetDownSample
import torchvision.transforms as T

def find_dataset_mean_std(dataset="cifar10"):
    available_datasets = ["cifar10", "cifar100", "food101", "imagenet", "mini", "imagenet32", "road_sign", "inat"]
    assert dataset in available_datasets, "Codes are not available for the selected dataset"
    mean = None
    std = None
    if dataset == "cifar10":
        mean = [0.491, 0.482, 0.446]
        std = [0.247, 0.243, 0.261]
    elif dataset == "cifar100":
        mean = [0.507, 0.487, 0.441]
        std = [0.262, 0.251, 0.271]
    elif dataset == "food101":
        mean = [0.550, 0.445, 0.340]
        std = [0.266, 0.269, 0.274]
    elif dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == "mini":
        mean = [0.471, 0.450, 0.404]
        std = [0.269, 0.261, 0.277]
    elif dataset == "imagenet32":
        mean = [0.481, 0.457, 0.408]
        std = [0.252, 0.245, 0.261]
    elif dataset == "road_sign":
        mean = [0.563, 0.567, 0.604]
        std = [0.338, 0.281, 0.345]
    elif dataset == "inat":
        mean = [0.467, 0.482, 0.376]
        std = [0.234, 0.225, 0.243]    
    return mean, std


def init_pretrain_transform(mean, std):
    train_transform = transforms.Compose([ 
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    val_transform = transforms.Compose([    
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
        ])

    return train_transform, val_transform

def init_dataset(train_transform, val_transform, dataset="cifar10"):
    available_datasets = ["cifar10", "cifar100", "food101", "imagenet", "mini", "imagenet32", "road_sign", "inat"]
    assert dataset in available_datasets, "Codes are not available for the selected dataset"
    train_dataset = None
    val_dataset = None
    num_classes = 0
    if dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=False,
                                            download=True, 
                                            transform=val_transform)
        num_classes = 10
    elif dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', 
                                            train=False,
                                            download=True, 
                                            transform=val_transform)
        num_classes = 100
    elif dataset == "food101":
        train_dataset = datasets.ImageFolder(root="./data/food-101/train", transform=train_transform)
        val_dataset = datasets.ImageFolder(root="./data/food-101/valid", transform=val_transform)
        num_classes = 101
    elif dataset == "imagenet":
        train_dataset = datasets.ImageFolder(root="../ImageNet/train", transform=train_transform)
        val_dataset = datasets.ImageFolder(root="../ImageNet/val", transform=val_transform)
        num_classes = 1000
    elif dataset == "mini":
        train_dataset = datasets.ImageFolder(root="../ImageNet/mini/train", transform=train_transform)
        num_classes = 100
    elif dataset == "imagenet32":
        train_dataset = ImageNetDownSample(root='../imagenet32/', 
                                            train=True,
                                            transform=train_transform)
        val_dataset = ImageNetDownSample(root='../imagenet32/', 
                                        train=False,
                                        transform=val_transform)
        num_classes = 1000
    elif dataset == "road_sign":
        train_dataset = datasets.ImageFolder(root='../road_sign/train', 
                                            transform=train_transform)
        val_dataset = datasets.ImageFolder(root='../road_sign/test', 
                                           transform=val_transform)
        num_classes = 8
    elif dataset == "inat":
        train_dataset = datasets.ImageFolder(root='../iNat2021/train_mini', 
                                            transform=train_transform)
        val_dataset = datasets.ImageFolder(root='../iNat2021/val', 
                                           transform=val_transform)
        num_classes = 10000
    return train_dataset, val_dataset, num_classes

def divide_dataset(train_dataset, ratio, dataset):
    available_datasets = ["cifar10", "cifar100", "food101", "imagenet", "mini", "imagenet32", "road_sign", "inat"]
    assert dataset in available_datasets, "Codes are not available for the selected dataset"
    train_size = len(train_dataset)
    labels = []
    if dataset == "cifar10" or dataset == "cifar100":
        labels = train_dataset.targets
    elif dataset == "food101" or dataset == "imagenet" or dataset == "mini" or dataset == "road_sign" or dataset == "inat":
        imgs = np.array(train_dataset.imgs)
        labels = imgs[:,1]
    else:
        labels = train_dataset.train_labels
    super_train_idxs, unsuper_train_idxs, _, _ = train_test_split(
        range(train_size),
        labels,
        stratify=labels,
        train_size=ratio,
        random_state=0
    )  
    return super_train_idxs, unsuper_train_idxs

def get_super_loaders(train_dataset, val_dataset, super_train_idxs, batch_size):

    super_train_dataset = torch.utils.data.Subset(train_dataset, super_train_idxs)

    supervised_dataloaders = {}
    
    supervised_dataloaders['train'] = torch.utils.data.DataLoader(super_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    supervised_dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return supervised_dataloaders

def get_unsuper_loaders(train_dataset, val_dataset, unsuper_train_idxs, clientIDlist, batch_size):

    unsupervised_dataloaders = {}
    step = 0
    train_step_size = int(len(unsuper_train_idxs) / len(clientIDlist))
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    for clientID in clientIDlist:
        unsupervised_dataloaders[clientID] = {}
        c_train_idxs = unsuper_train_idxs[step*train_step_size : (step+1)*train_step_size]
        c_trainset = torch.utils.data.Subset(train_dataset, c_train_idxs)
        c_trainloader = torch.utils.data.DataLoader(c_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        unsupervised_dataloaders[clientID]['train'] = c_trainloader
        unsupervised_dataloaders[clientID]['val'] = valloader
        step += 1

    return unsupervised_dataloaders

def get_unsuper_datasets(train_dataset, unsuper_train_idxs, clientIDlist,  sampling="iid", alpha=0.1, dataset="cifar10", num_classes=10, pre_transforms=None):
    check_dataset_valid(dataset)
    step = 0
    unsupervised_datasets = {}
    num_clients = len(clientIDlist)
    train_size = len(train_dataset)
    if dataset == "cifar10" or dataset == "cifar100":
        all_ids_train = np.array(train_dataset.targets)
    elif dataset == "food101" or dataset == "imagenet" or dataset == "mini" or dataset == "road_sign" or dataset == "inat":
        imgs = np.array(train_dataset.imgs)
        all_ids_train = [int(imgs[i, 1]) for i in range(imgs.shape[0])]
        all_ids_train = np.array(all_ids_train)
    else:
        all_ids_train = np.array(train_dataset.train_labels)

    if sampling == "iid":
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True)
        for i, (_, c_idxs) in enumerate(skf.split(np.zeros(train_size), all_ids_train)):
            clientID = clientIDlist[i]
            c_trainset = torch.utils.data.Subset(train_dataset, c_idxs)
            unsupervised_datasets[clientID] = c_trainset
    elif sampling == "dir":  # dir Label-skew
        class_ids_train = {class_num: np.where(all_ids_train == class_num)[0] for class_num in range(num_classes)}
        dist_of_client = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_classes).transpose()
        dist_of_client /= dist_of_client.sum()

        for i in range(100):
            s0 = dist_of_client.sum(axis=0, keepdims=True)
            s1 = dist_of_client.sum(axis=1, keepdims=True)
            dist_of_client /= s0
            dist_of_client /= s1
        
        samples_per_class_train = (np.floor(dist_of_client * train_size))

        start_ids_train = np.zeros((num_clients + 1, num_classes), dtype=np.int32)
        for i in range(0, num_clients):
            start_ids_train[i+1] = start_ids_train[i] + samples_per_class_train[i]

        for client_num in range(num_clients):
            l = np.array([], dtype=np.int32)
            for class_num in range(num_classes):
                start, end = start_ids_train[client_num, class_num], start_ids_train[client_num + 1, class_num]
                l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
            c_trainset = torch.utils.data.Subset(train_dataset, l)
            client_ID = clientIDlist[client_num]
            unsupervised_datasets[client_ID] = c_trainset

    elif sampling == "shard":  # simple Label-skew
        idxs = np.arange(len(all_ids_train))
        labels = all_ids_train

        # Sort by label
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # Divide into shards
        num_shards_total = num_clients * num_shards
        shard_size = len(idxs) // num_shards_total
        shards = [idxs[i*shard_size:(i+1)*shard_size] for i in range(num_shards_total)]
        np.random.shuffle(shards)

        for client_num in range(num_clients):
            client_shards = shards[client_num*num_shards:(client_num+1)*num_shards]
            client_idxs = np.concatenate(client_shards)
            client_ID = clientIDlist[client_num]
            c_trainset = torch.utils.data.Subset(train_dataset, client_idxs)
            unsupervised_datasets[client_ID] = c_trainset

    elif sampling == "feature_skew":  # Feature-skew 

        assert pre_transforms

        clear_transform = T.Compose([
            T.ToTensor(),
        ])

        train_dataset, _, num_classes = init_dataset(clear_transform, clear_transform, dataset=pre_transforms[1])

        # Basic even split of indices (like iid)
        train_step_size = int(len(unsuper_train_idxs) / len(clientIDlist))
        for i, clientID in enumerate(clientIDlist):
            c_train_idxs = unsuper_train_idxs[i*train_step_size : (i+1)*train_step_size]

            # Assign each client a slightly different augmentation
            transform = T.Compose([
                T.ToTensor(),
                T.RandomApply([T.ColorJitter(brightness=0.5 * np.random.rand())], p=0.8),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                T.RandomHorizontalFlip(),
                T.ToPILImage(),
            ])

            # Apply transform through a custom dataset wrapper
            c_trainset = FeatureSkewedDataset(train_dataset, c_train_idxs, feature_transform=transform, pre_transform=pre_transforms[0])
            unsupervised_datasets[clientID] = c_trainset

    else:
        raise ValueError(f"Sampling type '{sampling}' is not supported.")

    return unsupervised_datasets

class FeatureSkewedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, feature_transform=None, pre_transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pre_transform = pre_transform
        self.feature_transform = feature_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_dataset[self.indices[idx]]

        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image)

        if self.feature_transform:
            image = self.feature_transform(image)

        if self.pre_transform:
            image = self.pre_transform(image)

        return image, label

def check_dataset_valid(dataset):
    available_datasets = ["cifar10", "cifar100", "food101", "imagenet", "mini", "imagenet32", "road_sign", "inat"]
    assert dataset in available_datasets, "Codes are not available for the selected dataset"

def get_super_dataset(train_dataset, super_train_idxs):

    super_train_dataset = torch.utils.data.Subset(train_dataset, super_train_idxs)

    return super_train_dataset

def init_finetune_transform(args, mean, std):

    train_transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation='bicubic',
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=mean,
        std=std
    )

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)

    val_transform = transforms.Compose([  
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, val_transform 


def init_client_distribution(train_dataset, unsuper_train_idxs, num_classes, c_classes, clientIDlist):
    num_per_class = unsuper_train_idxs //  num_classes
    num_per_client = unsuper_train_idxs // len(clientIDlist)
    num_per_class_in_client = num_per_client // c_classes
    num_client_w_class = num_per_class // num_per_class_in_client
    class_counter = {i : num_client_w_class for i in range(num_classes)}
    total_counter = num_client_w_class * num_classes
    client_classes_dict = {}
    class_clients_dict = {i : [] for i in range(num_classes)}
    class_list = [i for i in range(num_classes)]
    for id in clientIDlist:
        pick_ok = False
        pick_classes = []
        client_classes_dict[id] = {}
        if total_counter < c_classes:
            for k, v in class_counter.item():
                if v > 0:
                    pick_classes.append(k)
        else:
            while not pick_ok:
                pick_ok = True
                pick_classes = random.sample(class_list, c_classes)
                for pc in pick_classes:
                    if class_counter[pc] == 0:
                        pick_ok = False
                        break
                    else:
                        class_counter[pc] = class_counter[pc] - 1
            total_counter  = total_counter - c_classes
        for pc in pick_classes:
            client_classes_dict[id][pc] = []
            class_clients_dict[pc].append(id)
    for idx in unsuper_train_idxs:
        idx_label = train_dataset[idx][1]
        allowed_clients = class_clients_dict[idx_label]
        if len(allowed_clients) > 0:
            pick_ok = False
            while not pick_ok:
                pick_ok = True
                random.shuffle(allowed_clients)
                pick_client = allowed_clients[0]
                if len(client_classes_dict[pick_client][idx_label]) < num_client_w_class:
                    client_classes_dict[pick_client][idx_label].append(idx)
                    if len(client_classes_dict[pick_client][idx_label]) == num_client_w_class:
                        allowed_clients.remove(pick_client)
                        class_clients_dict[idx_label] = allowed_clients
                else:
                    pick_ok = False
        else:
            set_ok = False
            for k in client_classes_dict.keys():
                if len(client_classes_dict[k].keys()) < c_classes:
                    if idx_label in client_classes_dict[k].keys():
                        client_classes_dict[k1][idx_label].append(idx)
                    else:
                        client_classes_dict[k1][idx_label] = [idx]
                    allowed_clients.append(k)
                    class_clients_dict[idx_label] = allowed_clients
                    set_ok = True
                    break
            if not set_ok:
                for k1 in client_classes_dict.keys():
                    count = 0
                    for k2 in client_classes_dict[k1].keys():
                        count += len(client_classes_dict[k1][k2])
                    if count < num_per_client:
                        if idx_label in client_classes_dict[k1].keys():
                            client_classes_dict[k1][idx_label].append(idx)
                        else:
                            client_classes_dict[k1][idx_label] = [idx]
    return client_classes_dict

def get_controlled_unsuper_datasets(train_dataset, client_classes_dict, clientIDlist):
    unsupervised_datasets = {}
    for clientID in clientIDlist:
        idxs = []
        for k in client_classes_dict[clientID].keys():
            idxs = idxs + client_classes_dict[clientID][k]
        c_trainset = torch.utils.data.Subset(train_dataset, idxs) 
        unsupervised_datasets[clientID] = c_trainset
    return unsupervised_datasets

def init_linprobe_transform(mean, std):

    train_transform = transforms.Compose([ 
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    val_transform = transforms.Compose([    
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    return train_transform, val_transform



    
        
            


    



