import os
import yaml
import random 
import shutil
import pandas as pd
from PIL import Image
random.seed(2025)

yolo_ds_path = '/home/ivan/Grounding-Dino-FineTuning/multimodal-data/YOLODataset'

save_dir = 'DINO_TinyTest'

# train_test_split = 0.8

images_path = os.path.join(yolo_ds_path, 'images')
labels_path = os.path.join(yolo_ds_path, 'labels')
meta_filepath = os.path.join(yolo_ds_path, 'dataset.yaml')

###
# Prepare new dataset folder
dst = '/'.join(yolo_ds_path.split('/')[:-1])
dst = os.path.join(dst, save_dir)
dst_im_train =os.path.join(dst, 'images/train') 
dst_im_val = os.path.join(dst, 'images/val')
os.makedirs(dst, exist_ok=True)
os.makedirs(dst_im_train, exist_ok=True)
os.makedirs(dst_im_val, exist_ok=True)

print(f'New dataset folder:\n{dst}\n')

###
# Read YOLO dataset classes
print('Reading YOLO classes:')

assert os.path.isfile(meta_filepath), f'Not found:\n{meta_filepath}'

with open(meta_filepath) as f:
    yamldata = yaml.safe_load(f)


class_map = yamldata['names']
print(f'{class_map}\n')

all_filenames = {
    'train':os.listdir(os.path.join(labels_path, 'train')),
    'val':os.listdir(os.path.join(labels_path, 'val'))
}

# random.shuffle(all_filenames)
# train_split = int(len(all_filenames) * train_test_split)

def data_from_filenames(filenames, task):
    data = []
    for name in filenames:
        filepath = os.path.join(labels_path, task, name)

        imgname = '.'.join(name.split('.')[:-1]) + '.png'
        imgpath = os.path.join(images_path, task, imgname)
        assert os.path.isfile(imgpath), f'File not found:\n{imgpath}'
        #print(f'imgpath: {imgpath}')
        width, height = Image.open(imgpath).size

        with open(filepath, 'r') as f:
            for line in f:
                # Split each line based on spaces
                parts = line.strip().split()
                assert len(parts)==5, "Each line should have 5 values"
                #print(parts)
                
                class_id = int(parts[0])
                assert class_id < len(class_map), 'Class ID out of bounds'
                class_name = class_map[class_id]

                x_center = int(float(parts[1]) * width)
                y_center = int(float(parts[2]) * height)
                bbox_w   = int(float(parts[3]) * width)
                bbox_h   = int(float(parts[4]) * height)
                
                x_top_left = int(x_center - (bbox_w / 2))
                y_top_left = int(y_center - (bbox_h / 2))

                row = [class_name, x_top_left, y_top_left, bbox_w, bbox_h, imgname, width, height]
                data.append(row)
    return data


###
# Write DINO dataset
columns = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'width', 'height']

def cp_imgs(path, data, task):
    for row in data:
        _src = os.path.join(images_path, task, row[5])
        # _dst = os.path.join(path, row[5])
        _dst = os.path.join(dst, f'images/{task}', row[5]) 
        if not os.path.isfile(_dst):
            shutil.copy(_src, _dst)

# import pdb;pdb.set_trace()
for task in ['train', 'val']:
    output_csv = os.path.join(dst, f'{task}_annotations.csv')
    
    _data = data_from_filenames(all_filenames[task], task)
    cp_imgs(dst_im_train, _data, task)

    df = pd.DataFrame(_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f'Annotations saved to {output_csv}')

print('done.')

