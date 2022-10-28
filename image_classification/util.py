# -*- coding: utf-8 -*-

import re
import json
import logging
import numpy as np
import os
import random
import torch
from PIL import Image

from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm, trange
from typing import List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

error_image = ["cloth_store_1416238_woman_19_0", "cloth_store_1416238_woman_4_8", "cloth_store_1416238_woman_20_6", "m_cloth_store_1416238_woman_19_0", "m_cloth_store_1416238_woman_4_8", "m_cloth_store_1416238_woman_20_6"]

@dataclass(frozen=True)
class image_data:
    image_name: str
    bbox: List[int]
    color: Optional[Union[List[str], str]]
    image_type: Optional[str]
    sleeveLength: Optional[str]
    pattern: Optional[str]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        logging.info('No GPU available, using the CPU instead.')

def load_data(data_path):
    samples = list()
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            if d["image_name"] not in error_image:
                samples.append(
                    image_data(
                        image_name = d["image_name"],
                        bbox = d["bbox"],
                        color = d.get("color"),
                        image_type = d.get("type"),
                        sleeveLength = d.get("sleeveLength"),
                        pattern = d.get("pattern")
                    )
                )

    return samples

def convert_samples_to_features(args, samples, extractor, label2idx_dict, eval=False, data_type="public"):
    image_path = dict()
    if data_type != "teststd":
        for p in os.listdir(os.path.join(args.image_data_dir, 'simmc2_scene_images_dstc10_public_part1')):
            image_path[p] = os.path.join(args.image_data_dir, 'simmc2_scene_images_dstc10_public_part1', p)
        for p in os.listdir(os.path.join(args.image_data_dir, 'simmc2_scene_images_dstc10_public_part2')):
            image_path[p] = os.path.join(args.image_data_dir, 'simmc2_scene_images_dstc10_public_part2', p)
    else:
        for p in os.listdir(os.path.join(args.image_data_dir, 'simmc2_scene_images_dstc10_teststd')):
            image_path[p] = os.path.join(args.image_data_dir, 'simmc2_scene_images_dstc10_teststd', p)
        
    scences = dict()
    for i in tqdm(image_path, desc='Image Load...'):
        if args.domain == 'fashion':
            if 'cloth' in i:
                try:
                    scences[i] = Image.open(image_path[i]).convert('RGB')
                except:
                    continue
        else:
            if 'wayfair' in i:
                try:
                    scences[i] = Image.open(image_path[i]).convert('RGB')
                except:
                    continue

    image_path.clear()
    label_dict = defaultdict(list)
    image = list()
    for sample in tqdm(samples, desc="Converting..."):
        scene = scences[sample.image_name]
        x, y, height, width = sample.bbox
        obj_img = scene.crop((x, y, x+width, y+height))
        values = extractor(obj_img, return_tensors="pt")["pixel_values"]
        if values.shape == (1, 3, 384, 384):
            image.append(values)
        else:
            import pdb; pdb.set_trace()

        if not eval:
            label_dict['type'].append(label2idx_dict['type'][sample.image_type])
            if args.domain == 'fashion':
                label_dict['sleeveLength'].append(label2idx_dict['sleeveLength'][sample.sleeveLength])
                label_dict['pattern'].append(label2idx_dict['pattern'][sample.pattern])
                label_dict['color'].append([1.0 if c in sample.color else 0.0 for c in label2idx_dict['color']])
            else:
                label_dict['color'].append(label2idx_dict['color'][sample.color])

    scences.clear()

    pixel_values = torch.cat(image, dim=0)

    if eval:
        dataset = TensorDataset(pixel_values)
    else:
        label_dict['type'] = torch.tensor(label_dict['type'], dtype=torch.long)
        if args.domain == 'fashion':
            label_dict['sleeveLength'] = torch.tensor(label_dict['sleeveLength'], dtype=torch.long)
            label_dict['pattern'] = torch.tensor(label_dict['pattern'], dtype=torch.long)
            label_dict['color'] = torch.tensor(label_dict['color'], dtype=torch.float)
            dataset = TensorDataset(pixel_values, label_dict['type'], label_dict['color'], label_dict['pattern'], label_dict['sleeveLength'])
        else:
            label_dict['color'] = torch.tensor(label_dict['color'], dtype=torch.long)
            dataset = TensorDataset(pixel_values, label_dict['type'], label_dict['color'])

    return dataset
