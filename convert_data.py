"""Convert image features from bottom up attention to numpy array"""

# Example
# python convert_data.py --imgid_list 'img_id.txt' --input_file 'test.csv' --output_file 'test.npy'

import os
import base64
import csv
import sys
import zlib
import json
import argparse
import h5py

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--imgid_list', default='data/coco_precomp/train_ids.txt',
                    help='Path to list of image id')
parser.add_argument('--input_file',
                    default='/media/data/kualee/coco_bottom_up_feature/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv',
                    help='tsv of all image data (output of bottom-up-attention/tools/generate_tsv.py), \
                    where each columns are: [image_id, image_w, image_h, num_boxes, boxes, features].')
parser.add_argument('--output_file', default='test.npy',
                    help='Output file path. the file saved in npy format')

opt = parser.parse_args()
print(opt)

meta = []
feature = {}
h = {}
w = {}
box = {}
for line in open(opt.imgid_list):
    sid = int(line.strip())
    meta.append(sid)
    feature[sid] = None

    h[sid] = None
    w[sid] = None
    box[sid] = None

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

if __name__ == '__main__':

    with open(opt.input_file, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                data = item[field]
                # buf = base64.decodestring(data)
                buf = base64.b64decode(data[1:])
                if field == 'boxes':
                    temp = np.frombuffer(buf, dtype=np.float64)
                if field == 'features':
                    temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item['num_boxes'], -1))
            if item['image_id'] in feature:
                feature[item['image_id']] = item['features']
            if item['image_id'] in h:
                h[item['image_id']] = item['image_h']
            if item['image_id'] in w:
                w[item['image_id']] = item['image_w']
            if item['image_id'] in box:
                box[item['image_id']] = item['boxes']
    data_out = np.stack([feature[sid] for sid in meta], axis=0)
    data_h = np.stack([h[sid] for sid in meta], axis=0)
    data_w = np.stack([w[sid] for sid in meta], axis=0)
    data_box = np.stack([box[sid] for sid in meta], axis=0)

    # Open the HDF5 file in write mode
    with h5py.File('visdial_val_features.h5', 'w') as file:
        # Create a dataset in the file and save the data
        file.create_dataset('image_ids', data=meta)
        file.create_dataset('boxes', data=data_box)
        file.create_dataset('features', data=data_out)

    with h5py.File('visdial_val_features.h5', 'r') as file:
        # Access datasets and attributes in the HDF5 file
        # For example, to access a dataset named 'data1':
        dataset1 = file['boxes']
        for i in dataset1:
            print(i)
            break
        print(dataset1)
    print("Final numpy array shape:", data_out.shape)
    np.save(opt.output_file, data_out)
