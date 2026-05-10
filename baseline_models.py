import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class SimCLRTransform():
    def __init__(self, is_sup, mean, std, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)]),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        self.mode = is_sup

    def __call__(self, x):
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 
            
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out) 

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None, img_size=32):
        super(ResNet, self).__init__()
        self.train_sup = (num_classes > 0)
        self.in_planes = 64
        self.img_size = img_size
        if self.img_size == 32:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.output_dim = 512*block.expansion
        if(self.train_sup):
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.img_size != 32:
            out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(self.train_sup):
            out = self.linear(out)
        return out

def create_backbone(name, num_classes=10):
    if name == 'res18':
        net = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
    elif name == 'res34':
        net = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    elif name == 'res18-origin':
        net = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, img_size=224)
    return net

######### SimSiam model class #########
class MLPact(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPact, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, affine=True)

    def forward(self, x):
        out = F.relu(self.bn(self.linear(x))) 
        return out
        
# Projector
class projection_MLP_simsiam(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=512):
        super(projection_MLP_simsiam, self).__init__()
        self.output_dim = out_dim
        self.layer1 = MLPact(in_dim, hidden_dim)
        self.layer2 = MLPact(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer3_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = self.layer3_bn(self.layer3(self.layer2(self.layer1(x))))
        return x 

# Predictor 
class prediction_MLP_simsiam(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512): 
        super(prediction_MLP_simsiam, self).__init__()
        self.layer1 = MLPact(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

# SimSiam
class simsiam(nn.Module):
    def __init__(self, bbone_arch):
        super(simsiam, self).__init__()
        self.register_buffer("rounds_done", torch.zeros(1))
        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = projection_MLP_simsiam(self.backbone.output_dim, hidden_dim=256, out_dim=512)

        ### Predictor (should be defined last for divergence aware update)
        self.predictor = prediction_MLP_simsiam(in_dim=self.projector.output_dim, out_dim=self.projector.output_dim)
    
    def forward(self, x1, x2, x3=None, deg_labels=None):
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = - (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2

        return L


def create_transform(img_size, mean, std):
    train_transform = SimCLRTransform(False, mean, std, image_size=img_size)
    test_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(), 
        T.Normalize(mean, std)]
    )
    return train_transform, test_transform