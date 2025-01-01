import pandas as pd
import os
import random

x = pd.read_csv('ROSTER_10Oct2024.csv')

# ADNI_1 = pd.read_csv('Search_screen_csv/ADNI_1.csv')
# ADNI_Go = pd.read_csv('Search_screen_csv/ADNI_Go.csv')
# ADNI_2 = pd.read_csv('Search_screen_csv/ADNI_2.csv')
# ADNI_3 = pd.read_csv('Search_screen_csv/ADNI_3.csv')
# ADNI_4 = pd.read_csv('Search_screen_csv/ADNI_4.csv')

# everything2 = pd.concat([ADNI_1,ADNI_Go,ADNI_2,ADNI_3,ADNI_4])

# print(everything2)


# print(len(everything2['Subject ID'].unique()))

#print(ADNI_1['Description'][ADNI_1['Description'].str.contains('rage',case=False)].unique())
#print(ADNI_Go['Description'][ADNI_Go['Description'].str.contains('rage',case=False)].unique())
#print(ADNI_2['Description'][ADNI_2['Description'].str.contains('rage',case=False)].unique())
#print(ADNI_3['Description'][ADNI_3['Description'].str.contains('mpr|rage',case=False)].unique())
#print(ADNI_4['Description'][ADNI_4['Description'].str.contains('mpr|rage',case=False)].unique())
#print(len(x['RID'].unique()))

print(os.getcwd())

#print(os.listdir())

IC_ADNI_1 = pd.read_csv('Image_Collections_csv/ADNI_1_10_12_2024.csv')
IC_ADNI_Go = pd.read_csv('Image_collections_csv/ADNI_Go_10_12_2024.csv')
IC_ADNI_2 = pd.read_csv('Image_collections_csv/ADNI_2_10_12_2024.csv')
IC_ADNI_3 = pd.read_csv('Image_collections_csv/ADNI_3_10_12_2024.csv')
IC_ADNI_4 = pd.read_csv('Image_collections_csv/ADNI_4_10_12_2024.csv')

everything = pd.concat([IC_ADNI_1,IC_ADNI_Go,IC_ADNI_2,IC_ADNI_3,IC_ADNI_4])

# print(everything)

# print(x)

# print(everything['Group'].value_counts())

# subject_ids_mri = everything['Subject'].unique()

# subject_ids_all = x['PTID'].unique()

# print(len(subject_ids_mri))

# print(len(subject_ids_all))

# set_all = set(subject_ids_all)

# set_mri = set(subject_ids_mri)

# subject_ids_not_mri = list(set_all - set_mri)

# print(subject_ids_not_mri[0:4])


# print(len(x['RID'].unique()))

random.seed(42)

string_list = list(x['PTID'].unique())

train_size = int(0.7 * len(string_list))
val_size = int(0.15 * len(string_list))
test_size = len(string_list) - train_size - val_size

train_list = random.sample(string_list, train_size)

remaining = [s for s in string_list if s not in train_list]

val_list = random.sample(remaining, val_size)

test_list = [s for s in remaining if s not in val_list]

print(everything['Group'].value_counts(normalize=True))

print(IC_ADNI_1['Group'].value_counts(normalize=True))

print(everything[everything['Subject'].isin(train_list)]['Group'].value_counts(normalize=True))

print(everything[everything['Subject'].isin(val_list)]['Group'].value_counts(normalize=True))

print(everything[everything['Subject'].isin(test_list)]['Group'].value_counts(normalize=True))

# print(len(train_list))
# print(len(val_list))
# print(len(test_list))

# split_labels = ['Train' for id in train_list] + \
#     ['Validation' for id in val_list] + ['Test' for id in test_list]

# df = pd.DataFrame({'Split':split_labels, 'ID':train_list + val_list + test_list})

# df.to_csv('data_split.csv',index=False)

# df = pd.read_csv('data_split.csv')

# train_ids = df[df['Split'] == 'Train']['ID']
# test_ids = df[df['Split'] == 'Test']['ID']
# val_ids = df[df['Split'] == 'Validation']['ID']

# train_bools = IC_ADNI_1['Subject'].isin(train_ids)
# test_bools = IC_ADNI_1['Subject'].isin(test_ids)
# val_bools = IC_ADNI_1['Subject'].isin(val_ids)

# train_len = len(IC_ADNI_1['Subject'][train_bools])
# test_len = len(IC_ADNI_1['Subject'][test_bools])
# val_len = len(IC_ADNI_1['Subject'][val_bools])

#print(IC_ADNI_3[train_bools])

# df = pd.DataFrame(IC_ADNI_1['Subject'])

# print(df)

# df['RID'] = df['Subject'].str[-4:]
# df['RID'] = [str(int(string)) for string in df['RID']]

# df2 = IC_ADNI_1[['Group', 'Acq Date', 'Visit', 'Description','Image Data ID']]

# df = pd.concat([df,df2], axis=1)

# print(df)

# df.to_csv('Splitting_Data/Dallan_data.csv',index=False)