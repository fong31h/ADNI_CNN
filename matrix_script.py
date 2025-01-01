import warnings
warnings.filterwarnings("ignore")
import pydicom as dicom
import matplotlib.pylab as plt
from os import listdir, walk, path, makedirs
import pandas as pd
import re
import numpy as np
import sys
import time
import xml.etree.ElementTree as ET

pattern = re.compile(r'ADNI_.+_.+_.+_.+_.+[_].+_(?:.+).+_(\d+)_.+_.+')
pattern2 = re.compile(r'I\d{4,7}')
pattern3 = re.compile(r'3d_matrix')
pattern4 = re.compile(r'MCI|AD|CN|EMCI|LMCI|SMC')
pattern5 = re.compile(r'ADNI_.+_.+_.+_.+_.+[_].+_(?:.+).+_(\d+)_.+_.+_pm.npy')

main_directory = sys.argv[1]
meta_dir = path.join(main_directory, 'Meta Data')
image_dir = path.join(main_directory,'ADNI')

def get_in_order_df(directory):
    file_names = listdir(directory)
    in_order_df = pd.DataFrame({'Names':file_names,'Order':[int(re.findall(pattern,filename)[0]) for filename in file_names]})
    in_order_df = in_order_df.sort_values('Order').reset_index(drop=True)
    return in_order_df

def get_3d_matrix(directory, in_order_df):
    matrices = []
    for i in range(len(in_order_df)):
        image = directory + '/' + in_order_df['Names'][i]
        ds = dicom.dcmread(image)
        matrices.append(ds.pixel_array)
    matrix_3d = np.stack(matrices,axis=0)
    return matrix_3d

def make_dir(root, dir):
    dir_path = path.join(root, dir + '_Pixel_Matrices/')
    if not path.exists(dir_path):
        makedirs(dir_path)
    return dir_path

def save_indiv_pm(in_order_df, dir_path, root, dir):
    for i in range(len(in_order_df)):
        image = path.join(root, dir) + '/' + in_order_df['Names'][i]
        pm = dicom.dcmread(image).pixel_array 
        np.save(dir_path + in_order_df['Names'][i][:-4] + '_pm', pm)

def save_matrices(directory):
    for root, dirs, files in walk(directory):
        for dir in dirs:
            if re.search(pattern2, dir):
                dir_path = make_dir(root, dir)
                #print(f'{path.join(root, dir)}')
                in_order_df = get_in_order_df(path.join(root, dir))
                matrix_3d = get_3d_matrix(path.join(root, dir), in_order_df)
                np.save(dir_path + dir + '_3d_matrix', matrix_3d)
                #print(f'Saved {dir} matrix')
                save_indiv_pm(in_order_df, dir_path, root, dir)

def populate_labels(paths_list, labels_list):
    labels = []
    length = len(paths_list)
    for i in range(length):
        image_id = re.findall(pattern2, paths_list[i])[0]
        y_entry = [entry for entry in labels_list if re.search(image_id, entry)][0]
        label = re.search(pattern4, y_entry)[0]
        labels.append(label)
    if len(labels) == length:
        df = pd.DataFrame({'File Path':paths_list, 'Research Group':labels})
        for i in range(len(df)):
            if (df['Research Group'][i] == 'EMCI') | (df['Research Group'][i] == 'LMCI'):
                df['Research Group'][i] = 'MCI'
            elif df['Research Group'][i] == 'SMC':
                df['Research Group'][i] = 'AD'
    else:
        print('Lengths incompatible')
    return df

def create_csv(directory1, directory2):
    paths_2d = []
    paths_3d = []
    groups = []
    labels = []
    for root, dirs, files in walk(directory1):
        for name in files:
            a = 'No'
            new_path = path.join(root, name)
            if name == '.DS_Store':
                continue
            if re.search(pattern3, name):
                paths_3d.append(new_path)
                a = 'Yes'
            if re.search(pattern5, name):
                paths_2d.append(new_path)
            if a == 'Yes':
                image_id = re.search(pattern2, name)
                for file in listdir(directory2):
                    if re.search(image_id.group(), file):
                        xml_file = path.join(directory2,file)
                        tree = ET.parse(xml_file)
                        xml_root = tree.getroot()
                        groups.append(xml_root.find('project').find('subject').find('researchGroup').text + file)
                        break
    for i in range(len(groups)):                
        label = re.search(pattern4, groups[i])[0]
        labels.append(label)          
    df1 = pd.DataFrame({'File Path':paths_3d, 'Research Group': labels})
    df1.to_csv(path.join(directory1, '3D_Images.csv'),index=False)
    df2 = populate_labels(paths_2d, groups)
    df2.to_csv(path.join(directory1, '2D_Images.csv'),index=False)

if __name__ == "__main__":
    if not path.isdir(meta_dir):
        print(f"Error: Required directory '{meta_dir}' does not exist.")
        sys.exit(1)
    print(f"Directory '{meta_dir}' exists. Continuing with the script...")
    print('This script will create 3D matrices, 2D matrices, and create two CSV files with the paths and labels for the respective dimensions.')
    start_time = time.time()
    save_matrices(image_dir)
    print('Created Matrices')
    create_csv(image_dir,meta_dir)
    print('Created CSV Files')
    end_time = time.time()
    print(f'{(end_time - start_time)/60} minutes')