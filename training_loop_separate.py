import time
from os import path, mkdir, listdir, remove 
import importlib
import Import_classes
importlib.reload(Import_classes)
import sys
from torchvision import transforms
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.models import ResNet50_Weights, resnet50
import torch.nn as nn
import torch.optim as optim
import re

# Create log file for stdout
stamp = time.time()
timeobject = time.localtime(stamp)
datetime = f'{timeobject[0]}-{timeobject[1]}-{timeobject[2]}'
log_path = 'Training_logs/'
if not path.exists(log_path):
    mkdir(log_path)
output_file = path.join(log_path, f'ADNI_1_training_loop.py_{datetime}.log')

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

training_loop_directory = input('Enter desired training loop directory (containing ADNI and "Meta Data" directories): \n')

# print(training_loop_directory)
# print(type(training_loop_directory))

transforms = transforms.Compose([Import_classes.NormalizeMatrix(), Import_classes.ResizeImage_2D()])

train_csv = path.join(training_loop_directory, f'ADNI/{input('Enter desired training data set (stored in ADNI/ folder): \n')}')
validation_csv = path.join(training_loop_directory, f'ADNI/{input('Enter desired validation data set (stored in ADNI/ folder): \n')}')
test_csv = path.join(training_loop_directory, f'ADNI/{input('Enter desired test data set (stored in ADNI/ folder): \n')}')

# print(training_csv)
# print(type(training_csv))

# Create the dataset
train_dataset = Import_classes.Rage_Scans(csv_file=train_csv,transform=transforms)
validation_dataset = Import_classes.Rage_Scans(csv_file=validation_csv,transform=transforms)
test_dataset = Import_classes.Rage_Scans(csv_file=test_csv,transform=transforms)

# Create training, testing, and validation sets
torch.manual_seed(input('Enter desired seed for reproducibility (between -9,223,372,036,854,775,808 and 18,446,744,073,709,551,615): \n'))

# Get batch size and num_workers
train_batch_size = int(input('Enter train batch size (rec: 256): \n'))
validation_batch_size = int(input('Enter validation batch size (rec: same as training): \n'))
test_batch_size = int(input('Enter test batch size (rec: same as training): \n'))
train_num_workers = int(input('Enter test num_workers (rec: 8): \n'))
validation_num_workers = int(input('Enter validation num_workers (rec: same as training): \n'))
test_num_workers = int(input('Enter test num_workers (rec: same as training): \n'))

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_num_workers)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=validation_num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=test_num_workers)

print('Loading in Resnet50 pretrained on ImageNet...')

# load in pretrained Resnet50
weights = ResNet50_Weights.IMAGENET1K_V1
resnet_50 = resnet50(weights=weights)

# set the first layer to accept 1 channel (grayscale) instead of 3. Not super sure of all the details going on.
original_conv1 = resnet_50.conv1
resnet_50.conv1 = nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                           stride=original_conv1.stride, padding=original_conv1.padding, bias=False)

with torch.no_grad():
    resnet_50.conv1.weight = nn.Parameter(original_conv1.weight.sum(dim=1, keepdim=True))


# set number of classes to 3
num_ftrs = resnet_50.fc.in_features
resnet_50.fc = nn.Linear(num_ftrs, 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_50 = resnet_50.to(device)

print('Resnet50 loaded in and modified to fit task.')
time.sleep(2)

print('Ready to begin training loop...')
time.sleep(2)

model_name = input('Enter desired model name (without suffix): \n')

# Freezing all layers but the final
for param in resnet_50.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer
resnet_50.fc.weight.requires_grad = True
resnet_50.fc.bias.requires_grad = True

# Setting the loss function
loss = nn.CrossEntropyLoss()

# Get learning rate
lr = float(input('Enter desired learning rate (rec: .001): \n'))
# Setting the optimizer
optimizer = optim.Adam(resnet_50.fc.parameters(), lr=lr)

print('Freezing layers')
time.sleep(1)
print('Adam optimizer and Cross Entropy loss selected')
time.sleep(1)

# Set epochs
epochs = int(input('Enter desired number of epochs (integer): \n'))
num_epochs = epochs
start_time = time.time()

# Early stopping parameters
patience = int(input('Entire desired patience parameter \n'))
min_delta = 0.001
best_val_loss = float('inf')
counter = 0

print('Beginning training...')

# Training loop
for epoch in range(num_epochs):
    resnet_50.train()
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        matrix = batch['matrix'].to(device, dtype=torch.float32)
        label = batch['label'].to(device, dtype=torch.long)
        matrix = torch.unsqueeze(matrix, 1)
        optimizer.zero_grad()
        outputs = resnet_50(matrix)
        
        batch_loss = loss(outputs, label)
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item()
        
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], '
                  f'Loss: {running_loss / 10:.4f}')
            running_loss = 0.0  # Reset running_loss after each print
    
    # Validation phase
    resnet_50.eval()
    val_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_dataloader):
            val_matrix = batch['matrix'].to(device, dtype=torch.float32)
            val_label = batch['label'].to(device, dtype=torch.long)
            val_matrix = torch.unsqueeze(val_matrix, 1)
            val_outputs = resnet_50(val_matrix)
            _, val_predicted = torch.max(val_outputs, 1)
            val_total_samples += val_label.size(0)
            val_total_correct += (val_predicted == val_label).sum().item()
            val_batch_loss = loss(val_outputs, val_label)
            val_loss += val_batch_loss.item()
    
    avg_val_loss = val_loss / len(validation_dataloader)
    val_accuracy = val_total_correct / val_total_samples

    print(f'Epoch [{epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.4f} Val Accuracy: {val_accuracy:.4f}')
    
    # Early stopping check
    if avg_val_loss < best_val_loss - min_delta:
        print('Model Updated')
        best_val_loss = avg_val_loss
        counter = 0
        for file in listdir('Models/'):
            if re.search(model_name, file):
                file_path = path.join('Models', file)
                remove(file_path)
                break
        torch.save(resnet_50.state_dict(), f'Models/{model_name}_epoch_{epoch + 1}.pth')  # Save the best model
    else:
        print('Model Not Updated')
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            for file in listdir('Models/'):
                if re.search(model_name, file):
                    file_path = path.join('Models', file)
                    break
            resnet_50.load_state_dict(torch.load(file_path))  # Load the best model
            break

print('Training complete')
end_time = time.time()
print(f'Counter value:{counter}')
print(f'Total training time: {(end_time - start_time)/3600} hours')
sys.stdout = sys.__stdout__
f.close()