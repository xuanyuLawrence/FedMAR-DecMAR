import torch
import torchvision
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from imagenetLoad import ImageNetDownSample

def get_argparser():
    parser = argparse.ArgumentParser(description="Find mean and std of dataset")
    parser.add_argument('-p', '--path', default='',
                        help='path of image folder')
    parser.add_argument('-n', '--name', default='', 
                        help='name of pytorch dataset')             
    return parser

def main(args):
    
    assert args.path or args.name, "Must give either a path or a name"
    
    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    image_data = None
    if args.path:
        image_data = torchvision.datasets.ImageFolder(
            root=args.path, transform=transform_img
        )
    else:
        names = ['cifar10', 'cifar100', 'imagenet32', "road_sign", "inat"]
        assert args.name in names, "The name of dataset is not allowed"
        if args.name == 'cifar10':
            image_data = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=transform_img)
        elif args.name == 'cifar100':
            image_data = torchvision.datasets.CIFAR100(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=transform_img)
        elif args.name == 'imagenet32':
            image_data = ImageNetDownSample(root='../imagenet32', 
                                            train=True,
                                            transform=transform_img)
        elif args.name == 'road_sign':
            image_data = torchvision.datasets.ImageFolder(root="../road_sign/train", transform=transform_img)
        elif args.name == 'inat':
            image_data = torchvision.datasets.ImageFolder(root="../iNat2021/train_mini", transform=transform_img)
        

    loader = DataLoader(image_data, batch_size=64, num_workers=1)

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)  
    print("mean and std: \n", mean, std)

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
