import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
from collections import Counter

# ----------------------------
# Configuration
# ----------------------------
seed = 10
np.random.seed(seed)

base_dir = './data/'  # Base project directory
csv_file = os.path.join(base_dir, 'ADNIMERGE.csv')
img_dir = os.path.join(base_dir, 'images/')
h5_noimg_path = os.path.join(base_dir, 'ADNI_longitudinal_noimg.h5')
h5_img_path = os.path.join(base_dir, 'ADNI_longitudinal_img.h5')
h5_aug_path = os.path.join(base_dir, 'ADNI_longitudinal_img_aug.h5')
npy_subj_path = os.path.join(base_dir, 'ADNI_longitudinal_subj.npy')
txt_output_dir = os.path.join(base_dir, 'splits/')
os.makedirs(txt_output_dir, exist_ok=True)

data_type = 'single'  # 'single' or 'pair'
aug_size = 10  # Number of augmentations per image

# ----------------------------
# Load subject-level metadata
# ----------------------------
df_raw = pd.read_csv(csv_file, usecols=['PTID', 'DX_bl', 'DX', 'EXAMDATE', 'MMSE',
                                        'Hippocampus', 'Ventricles', 'ABETA', 'TAU', 'AGE',
                                        'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'PTGENDER'])

# ----------------------------
# Helper: Load HDF5 to dict (if exists)
# ----------------------------
def h5_to_dict(h5file):
    """Recursively load HDF5 content into dictionary."""
    def recursive_dict(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursive_dict(item)
        return result

    if not os.path.exists(h5file):
        return {}
    with h5py.File(h5file, 'r') as f:
        return recursive_dict(f)

h5_data_dict = h5_to_dict(h5_noimg_path)
subj_h5 = list(h5_data_dict.keys())

# ----------------------------
# Prepare image and label data
# ----------------------------
label_dict = {'Normal': 0, 'NC': 0, 'CN': 0, 'MCI': 1, 'LMCI': 1, 'EMCI': 1,
              'AD': 2, 'Dementia': 2, 'sMCI': 3, 'pMCI': 4}

img_paths = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
subj_data = {}
nan_label_count = 0
nan_idx_list = []
gender_list = []

for img_path in img_paths:
    subj_id = os.path.basename(img_path).split('-')[0]
    date_str = '-'.join(os.path.basename(img_path).split('-')[1:4]).split('_')[0]
    date_struct = datetime.strptime(date_str, '%Y-%m-%d')
    rows = df_raw[df_raw['PTID'] == subj_id]

    if rows.shape[0] == 0:
        continue

    # Match closest exam date
    date_diff = [abs((datetime.strptime(d, '%Y-%m-%d') - date_struct).days) for d in rows['EXAMDATE']]
    i = np.argmin(date_diff)
    if date_diff[i] > 120:
        continue

    # Initialize subject dictionary
    if subj_id not in subj_data:
        subj_data[subj_id] = {'Hippocampus': [], 'Ventricles': [], 'ABETA': [], 'TAU': [],
                              'WholeBrain': [], 'Entorhinal': [], 'Fusiform': [], 'MidTemp': [],
                              'MMSE': [], 'label_all': [], 'AGE': [],
                              'label': label_dict[rows.iloc[i]['DX_bl']],
                              'date': [], 'date_start': date_struct,
                              'date_interval': [], 'img_paths': []}

    if rows.iloc[i]['EXAMDATE'] in subj_data[subj_id]['date']:
        continue

    # Record data
    subj_data[subj_id]['date'].append(rows.iloc[i]['EXAMDATE'])
    subj_data[subj_id]['date_interval'].append((date_struct - subj_data[subj_id]['date_start']).days / 365.)
    subj_data[subj_id]['img_paths'].append(os.path.basename(img_path))

    if pd.notnull(rows.iloc[i]['DX']):
        subj_data[subj_id]['label_all'].append(label_dict[rows.iloc[i]['DX']])
        for k in ['MMSE', 'Hippocampus', 'Ventricles', 'ABETA', 'TAU',
                  'AGE', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']:
            subj_data[subj_id][k].append(rows.iloc[i][k])
    else:
        nan_label_count += 1
        nan_idx_list.append([subj_id, len(subj_data[subj_id]['label_all'])])
        for k in ['label_all', 'MMSE', 'Hippocampus', 'Ventricles', 'ABETA', 'TAU',
                  'AGE', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']:
            subj_data[subj_id][k].append(-1)

# ----------------------------
# Fill missing labels
# ----------------------------
for subj, idx in nan_idx_list:
    for key in ['label_all', 'MMSE', 'Hippocampus', 'Ventricles', 'ABETA', 'TAU',
                'AGE', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']:
        subj_data[subj][key][idx] = subj_data[subj][key][idx-1]

# ----------------------------
# Classify subjects: NC, MCI, AD
# ----------------------------
subj_list_dict = {'NC': [], 'MCI': [], 'AD': []}

for subj_id, info in subj_data.items():
    labels = set(info['label_all'])
    if len(labels) != 1:
        if labels <= {1, 2} or labels == {0, 1, 2}:
            info['label'] = 2
            subj_list_dict['AD'].append(subj_id)
        elif labels == {0, 1}:
            info['label'] = 1
            subj_list_dict['MCI'].append(subj_id)
        elif labels == {0, 2}:
            info['label'] = 2
            subj_list_dict['AD'].append(subj_id)
    elif info['label'] == 1:
        subj_list_dict['MCI'].append(subj_id)
    elif info['label'] == 0:
        subj_list_dict['NC'].append(subj_id)
    else:
        subj_list_dict['AD'].append(subj_id)

np.save(npy_subj_path, subj_list_dict)

# ----------------------------
# Clean ABETA and TAU values
# ----------------------------
def clean_value(value):
    """Convert list with '<' or '>' to numeric."""
    if isinstance(value, list) and any(isinstance(x, str) for x in value):
        value = [v.replace('>', '').replace('<', '') if isinstance(v, str) else v for v in value]
        return [float(v) if str(v) != 'nan' else np.nan for v in value]
    return value

for subj in subj_data:
    subj_data[subj]['ABETA'] = clean_value(subj_data[subj]['ABETA'])
    subj_data[subj]['TAU'] = clean_value(subj_data[subj]['TAU'])

# ----------------------------
# Save non-image data
# ----------------------------
with h5py.File(h5_noimg_path, 'w') as f_noimg:
    for subj_id, info in subj_data.items():
        grp = f_noimg.create_group(subj_id)
        for key in ['label', 'label_all', 'date_interval', 'MMSE', 'Hippocampus',
                    'Ventricles', 'ABETA', 'TAU', 'AGE', 'WholeBrain',
                    'Entorhinal', 'Fusiform', 'MidTemp']:
            grp.create_dataset(key, data=info[key])

# ----------------------------
# Save images to HDF5
# ----------------------------
if not os.path.exists(h5_img_path):
    with h5py.File(h5_img_path, 'w') as f_img:
        for i, (subj_id, info) in enumerate(subj_data.items()):
            grp = f_img.create_group(subj_id)
            for img_name in info['img_paths']:
                img_nib = nib.load(os.path.join(img_dir, img_name))
                img = img_nib.get_fdata()
                img = (img - np.mean(img)) / np.std(img)
                grp.create_dataset(img_name, data=img)
            print(f"[{i+1}/{len(subj_data)}] Saved images for subject {subj_id}")

# ----------------------------
# Image augmentation
# ----------------------------
def augment_image(img, rotate, shift, flip):
    """Apply small rotations, shifts, and flips to 3D MRI."""
    img = scipy.ndimage.rotate(img, rotate[0], axes=(1, 0), reshape=False)
    img = scipy.ndimage.rotate(img, rotate[1], axes=(0, 2), reshape=False)
    img = scipy.ndimage.rotate(img, rotate[2], axes=(1, 2), reshape=False)
    img = scipy.ndimage.shift(img, shift[0])
    if flip[0] == 1:
        img = np.flip(img, 0)
    return img

if not os.path.exists(h5_aug_path):
    with h5py.File(h5_aug_path, 'w') as f_aug:
        for i, (subj_id, info) in enumerate(subj_data.items()):
            grp = f_aug.create_group(subj_id)
            for img_name in info['img_paths']:
                img_nib = nib.load(os.path.join(img_dir, img_name))
                img = img_nib.get_fdata()
                img = (img - np.mean(img)) / np.std(img)
                imgs = [img]
                rotate_list = np.random.uniform(-2, 2, (aug_size - 1, 3))
                shift_list = np.random.uniform(-2, 2, (aug_size - 1, 1))
                flip_list = np.random.randint(0, 2, (aug_size - 1, 1))
                for j in range(aug_size - 1):
                    imgs.append(augment_image(img, rotate_list[j], shift_list[j], flip_list[j]))
                grp.create_dataset(img_name, data=np.stack(imgs, 0))
            print(f"[{i+1}/{len(subj_data)}] Augmented images for {subj_id}")

# ----------------------------
# Dataset splitting (5-fold)
# ----------------------------
def save_txt(path, subj_ids, case_ids, mode='single'):
    with open(path, 'w') as f:
        for subj, case in zip(subj_ids, case_ids):
            if mode == 'single':
                f.write(f"{subj} {case[0]} {case[1]}\n")
            else:
                f.write(f"{subj} {case[0]} {case[1]} {case[2]} {case[3]}\n")

def get_single_case_list(subj_data, subj_ids):
    subj_list, case_list = [], []
    for subj in subj_ids:
        for i, case in enumerate(subj_data[subj]['img_paths']):
            subj_list.append(subj)
            case_list.append([case, i])
    return subj_list, case_list

def get_pair_case_list(subj_data, subj_ids):
    subj_list, case_list = [], []
    for subj in subj_ids:
        cases = subj_data[subj]['img_paths']
        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                subj_list.append(subj)
                case_list.append([cases[i], cases[j], i, j])
    return subj_list, case_list

subj_all = np.load(npy_subj_path, allow_pickle=True).item()

for fold in range(5):
    train, val, test = [], [], []
    for cls in ['NC', 'MCI', 'AD']:
        cls_list = subj_all[cls]
        np.random.shuffle(cls_list)
        n = len(cls_list)
        test_cls = cls_list[fold * int(0.2 * n):(fold + 1) * int(0.2 * n)]
        val_cls = cls_list[:int(0.1 * (n - len(test_cls)))]
        train_cls = [x for x in cls_list if x not in test_cls + val_cls]
        train += train_cls
        val += val_cls
        test += test_cls

    if data_type == 'single':
        train_subj, train_case = get_single_case_list(subj_data, train)
        val_subj, val_case = get_single_case_list(subj_data, val)
        test_subj, test_case = get_single_case_list(subj_data, test)
    else:
        train_subj, train_case = get_pair_case_list(subj_data, train)
        val_subj, val_case = get_pair_case_list(subj_data, val)
        test_subj, test_case = get_pair_case_list(subj_data, test)

    save_txt(os.path.join(txt_output_dir, f'fold{fold}_train.txt'), train_subj, train_case, data_type)
    save_txt(os.path.join(txt_output_dir, f'fold{fold}_val.txt'), val_subj, val_case, data_type)
    save_txt(os.path.join(txt_output_dir, f'fold{fold}_test.txt'), test_subj, test_case, data_type)

    print(f"Fold {fold}: Train={len(train)}, Val={len(val)}, Test={len(test)}")

print("✅ Preprocessing complete.")
