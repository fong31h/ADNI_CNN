import re
import pandas as pd

df = pd.read_csv('ADNI_1/ADNI/2D_Images_duplicated.csv')

split_df = pd.read_csv('data_split.csv')

train_ids = split_df[split_df['Split'] == 'Train']['ID']
test_ids = split_df[split_df['Split'] == 'Test']['ID']
validation_ids = split_df[split_df['Split'] == 'Validation']['ID']

pattern = re.compile(r'ADNI_1\/ADNI\/(\d{3}_S_\d+)\/')

columns = ['File Path', 'Research Group']

train_df = pd.DataFrame(columns=columns)

test_df = pd.DataFrame(columns=columns)

validation_df = pd.DataFrame(columns=columns)

train_paths, validation_paths, test_paths = [], [], []

train_groups, validation_groups, test_groups = [], [], []

train_ids_list = train_ids.to_list()
validation_ids_list = validation_ids.to_list()
test_ids_list = test_ids.to_list()


for i in range(len(df)):
    ID = re.search(pattern, df['File Path'][i])[1]
    if ID in train_ids_list:
        train_paths.append(df['File Path'][i])
        train_groups.append(df['Research Group'][i])
    elif ID in test_ids_list:
        test_paths.append(df['File Path'][i])
        test_groups.append(df['Research Group'][i])
    elif ID in validation_ids_list:
        validation_paths.append(df['File Path'][i])
        validation_groups.append(df['Research Group'][i])

train_df['File Path'] = train_paths
train_df['Research Group'] = train_groups

validation_df['File Path'] = validation_paths
validation_df['Research Group'] = validation_groups

test_df['File Path'] = test_paths
test_df['Research Group'] = test_groups

print(len(train_df), len(test_df), len(validation_df))

train_df.to_csv('ADNI_1/ADNI/train_data.csv', index=False)
validation_df.to_csv('ADNI_1/ADNI/validation_data.csv', index=False)
test_df.to_csv('ADNI_1/ADNI/test_data.csv', index=False)