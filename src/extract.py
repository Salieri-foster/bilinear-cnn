
import os
import pickle

import numpy as np
import PIL.Image
import torch


root ="D:/SRT/data/cub200"
image_path = os.path.join(root ,'raw/CUB_200_2011/images/')
# Format of images.txt: <image_id> <image_name>
id2name = np.genfromtxt(os.path.join(
    root ,'raw/CUB_200_2011/images.txt') ,dtype=str)
# Format of train_test_split.txt: <image_id> <is_training_image>
id2train = np.genfromtxt(os.path.join(
    root ,'raw/CUB_200_2011/train_test_split.txt') ,dtype=int)

train_data = [ ]
train_labels = [ ]
test_data = [ ]
test_labels = [ ]
for id_ in range(id2name.shape[ 0 ]):
    image = PIL.Image.open(os.path.join(image_path ,id2name[ id_ ,1 ]))
    label = int(id2name[ id_ ,1 ][ :3 ]) - 1  # Label starts with 0

    # Convert gray scale image to RGB image.
    if image.getbands()[ 0 ] == 'L':
        image = image.convert('RGB')
    image_np = np.array(image)
    image.close()

    if id2train[ id_ ,1 ] == 1:
        train_data.append(image_np)
        train_labels.append(label)
    else:
        test_data.append(image_np)
        test_labels.append(label)

pickle.dump((train_data ,train_labels) ,
               open(os.path.join(root ,'processed/train.pkl') ,'wb'))
pickle.dump((test_data ,test_labels) ,
               open(os.path.join(root ,'processed/test.pkl') ,'wb'))