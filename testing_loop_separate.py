import time
import sys
import torch
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms
import Import_classes
from os import path, mkdir
from torch.utils.data import random_split, DataLoader
import torch.nn as nn


# Create log file for stdout
stamp = time.time()
timeobject = time.localtime(stamp)
datetime = f'{timeobject[0]}-{timeobject[1]}-{timeobject[2]}'
log_path = 'Testing_logs/'
if not path.exists(log_path):
    mkdir(log_path)
output_file = path.join(log_path, f'testing_loop_separate.py_{datetime}.log')

# Redirect stdout to the file but still print to terminal
f = open(output_file, 'w')
both = Import_classes.Write_Both(f)
sys.stdout = both
stdin_redirector = Import_classes.StdInRedirector(f)

# Override the built-in input
original_input = input  # Save original input
input = stdin_redirector.input

print('Warning: Be sure this script is running from a base directory that contains the directory with data you want to work with.')
time.sleep(3)

testing_loop_directory = input('Enter desired testing loop directory (containing ADNI and "Meta Data" directories): \n')

model_input = input('This script is intended for use with a Resnet50 state dictionary modified for the ADNI MRI project. Please enter name of model desired. \n')

state_dict = torch.load(f'Models/{model_input}')
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)

# set the first layer to accept 1 channel (grayscale) instead of 3. Not super sure of all the details going on.
original_conv1 = model.conv1
model.conv1 = nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                           stride=original_conv1.stride, padding=original_conv1.padding, bias=False)

with torch.no_grad():
    model.conv1.weight = nn.Parameter(original_conv1.weight.sum(dim=1, keepdim=True))

# set number of classes to 3
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(state_dict)

transforms = transforms.Compose([Import_classes.NormalizeMatrix(), Import_classes.ResizeImage_2D()])

testing_csv = path.join(testing_loop_directory, f'ADNI/{input('Enter desired testing loop data set (stored in ADNI/ folder): \n')}')

# Create the dataset
test_dataset = Import_classes.Rage_Scans(csv_file=testing_csv,transform=transforms)

print('Warning: Be sure to use the same parameter inputs as used to train this model.')

# Dataloader inputs
torch.manual_seed(input('Enter seed for reproducibility (between -9,223,372,036,854,775,808 and 18,446,744,073,709,551,615): \n'))
test_batch_size = int(input('Enter test batch_size \n'))
test_num_workers = int(input('Enter test num_workers \n'))

# Create the dataloader
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=test_num_workers)

print('Beginning testing loop...')

model.eval()

total_correct = 0
total_samples = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        matrix = batch['matrix']
        labels = batch['label']
        matrix = torch.unsqueeze(matrix, 1)
        outputs = model(matrix)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        total_steps = len(test_dataloader)
        interval = max(1, total_steps // 20)  # Ensure the interval is at least 1
        if (i + 1) % interval == 0 or i == total_steps - 1:  # Print for every interval and the last step
            print(f'Step {i + 1} / {total_steps}')

print(f'Accuracy: {total_correct}/{total_samples} = {total_correct / total_samples}')