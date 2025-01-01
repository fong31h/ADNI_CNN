import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

# Rage_Scans Dataset Class

class Rage_Scans(Dataset):

    def __init__(self, csv_file, transform=None):
        
        self.data_frame = pd.read_csv(csv_file)
        unique_labels = ['CN', 'MCI', 'AD']
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        self.data_frame['Research Group'] = pd.Series([label_to_index[label] for label in self.data_frame['Research Group']])
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        matrix_path = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]

        matrix = np.load(matrix_path)
        matrix = matrix.astype('float32')

        sample = {'matrix': matrix, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class NormalizeMatrix:
    def __call__(self, sample):
        matrix, label = sample['matrix'], sample['label']
        # Normalize the matrix between 0 and 1
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
        return {'matrix': matrix, 'label': label}
    
class PadMatrix_3D:
    def __call__(self, sample):
        matrix, label = sample['matrix'], sample['label']
        if matrix.shape == (208, 256, 256):
            return sample
        
        dim1, dim2, dim3 = matrix.shape

        # Padding for the first dimension
        if dim1 != 208:
            padding = ((int((208 - dim1) / 2), int((208 - dim1) / 2)), (0, 0), (0, 0))
            matrix = np.pad(matrix, padding, mode='constant')
        
        # Padding for the second dimension
        if dim2 == 192:
            padding = ((0, 0), (32, 32), (32, 32))
            matrix = np.pad(matrix, padding, mode='constant')
            return {'matrix': matrix, 'label': label}

        # Padding for the third dimension
        if dim3 == 240:
            padding = ((0, 0), (0, 0), (8, 8))
            matrix = np.pad(matrix, padding, mode='constant')
        
        return {'matrix': matrix, 'label': label}

class ResizeImage_2D:
    def __call__(self, sample):
        image, label = sample['matrix'], sample['label']
        
        # Check if the image needs resizing
        if image.shape[0] < 256 or image.shape[1] < 256:
            # Convert the image to a PIL Image for resizing
            image = Image.fromarray(image)
            
            # Resize the image while maintaining the aspect ratio
            image = image.resize((256, 256), Image.BILINEAR)
            
            # Convert the PIL image back to a numpy array
            image = np.array(image)
        
        return {'matrix': image, 'label': label}

class pad_250_256:
    # This exists for my pixel importance work (not a permanent option
    # once that is optimized)
    def __call__(self, sample):
        matrix, label = sample['matrix'], sample['label']
        matrix = np.pad(matrix, ((0,6),(0,6)), constant_values=(0))
        return {'matrix':matrix, 'label': label}