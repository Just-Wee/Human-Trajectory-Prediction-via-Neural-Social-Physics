import os
import shutil

from tqdm import tqdm

src_path = '/home/liuhaijing/dataset/SDD/SDD/'
src_file_names = os.listdir(src_path)

dst_path = '/home/liuhaijing/Human-Trajectory-Prediction-via-Neural-Social-Physics/data/SDD/'

dst_train_pickle_path = dst_path + 'train_pickle_new/'
dst_test_pickle_path = dst_path + 'test_pickle_new/'
dst_none_pickle_path = dst_path + 'none_pickle_new/'
dst_train_mask_path = dst_path + 'train_masks/'
dst_test_mask_path = dst_path + 'test_masks/'

train_names = list(map(
    lambda n: n.split('_')[0]+'video'+n.split('_')[1],
    os.listdir(dst_train_mask_path)
))
test_names = list(map(
    lambda n: n.split('_')[0]+'video'+n.split('_')[1],
    os.listdir(dst_test_mask_path)
))

print(train_names[:10])
print(test_names[:10])

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

mkdir(dst_train_pickle_path)
mkdir(dst_test_pickle_path)
mkdir(dst_none_pickle_path)

for n in tqdm(src_file_names):
    name = n[:n.find('.')] 
    if name in train_names:
        shutil.copy(src_path + n, dst_train_pickle_path)
    elif name in test_names:
        shutil.copy(src_path + n, dst_test_pickle_path)
    else:
        shutil.copy(src_path + n, dst_none_pickle_path)
