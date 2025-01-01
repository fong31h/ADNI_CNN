import torch
from torchvision.models import ResNet50_Weights, resnet50

# load in pretrained Resnet50
weights = ResNet50_Weights.IMAGENET1K_V1
resnet_50 = resnet50(weights=weights)

# set the first layer to accept 1 channel (grayscale) instead of 3. Not super sure of all the details going on.
original_conv1 = resnet_50.conv1
resnet_50.conv1 = torch.nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                           stride=original_conv1.stride, padding=original_conv1.padding, bias=False)

with torch.no_grad():
    resnet_50.conv1.weight = torch.nn.Parameter(original_conv1.weight.sum(dim=1, keepdim=True))

# set device
device = 'cpu'

# set number of classes to 3
num_ftrs = resnet_50.fc.in_features
resnet_50.fc = torch.nn.Linear(num_ftrs, 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_50 = resnet_50.to(device)

torch.save(resnet_50, 'Models/base_resnet50.pth')