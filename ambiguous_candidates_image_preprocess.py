# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch

from collections import defaultdict
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm
from PIL import Image

error_image = ["cloth_store_1416238_woman_19_0", "cloth_store_1416238_woman_4_8", "cloth_store_1416238_woman_20_6", "m_cloth_store_1416238_woman_19_0", "m_cloth_store_1416238_woman_4_8", "m_cloth_store_1416238_woman_20_6"]

def image_embed(args, image_data, all_jsonname, data_type):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    bbox = defaultdict(dict)
    obj_image_embedding = defaultdict(dict)
    scene_image_embedding = dict()
    for scene_name in tqdm(all_jsonname, desc='image preprocessing...'):
        if scene_name in error_image:
            continue

        try:
            with open(os.path.join(args.scene_json_folder, '{}_scene.json'.format(scene_name)), 'r', encoding='utf-8') as f:
                scene_json = json.load(f)
        except:
            continue

        image_name = scene_name.replace('m_', '') + ".png"
        scene = Image.open(image_data[image_name]).convert('RGB')
        inputs = {k: v.to(device) for k, v in feature_extractor(images=scene, return_tensors="pt").items()}
        with torch.no_grad():
            outputs = model(**inputs)
        scene_image_embedding[scene_name] = outputs.last_hidden_state[0, 0].detach().cpu()

        w_img, h_img = scene.size

        for obj in scene_json['scenes'][0]['objects']:
            x, y, height, width = obj['bbox']
            obj_img = scene.crop((x, y, x+width, y+height))

            x_center = (x + width/2) / w_img
            y_center = (y - height/2) / h_img
            w = width / w_img
            h = height / h_img
            bbox[scene_name][obj['index']] = torch.tensor([x_center, y_center, w, h], dtype=torch.float)

            inputs = {k: v.to(device) for k, v in feature_extractor(images=obj_img, return_tensors="pt").items()}
            with torch.no_grad():
                outputs = model(**inputs)
            
            obj_image_embedding[scene_name][obj['index']] = outputs.last_hidden_state[0, 0].detach().cpu()

    if data_type == "train":
        bbox['no_answer'] = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)
        obj_image_embedding['no_answer'] = torch.zeros(768, dtype=torch.float)
        scene_image_embedding['no_answer'] = torch.zeros(768, dtype=torch.float)

    torch.save(bbox, os.path.join(args.save_folder, 'obj_image_{}_normalized_bbox.bin'.format(data_type)))
    torch.save(scene_image_embedding, os.path.join(args.save_folder, 'scene_image_{}.bin'.format(data_type)))
    torch.save(obj_image_embedding, os.path.join(args.save_folder, 'obj_image_{}.bin'.format(data_type)))


def main(args):
    if args.data_mode == 'public':
        image_data = dict()
        for p in os.listdir(args.scene_image_folder + '_part1'):
            image_data[p] = os.path.join(args.scene_image_folder + '_part1', p)
        for p in os.listdir(args.scene_image_folder + '_part2'):
            image_data[p] = os.path.join(args.scene_image_folder + '_part2', p)

        for data_type in ['train', 'dev', 'devtest']:
            all_jsonname = set()
            with open(os.path.join(args.data_folder, 'simmc2.1_dials_dstc11_{}.json'.format(data_type)), 'r') as f:
                dialogs = json.load(f)
                for dial in dialogs["dialogue_data"]:
                    all_jsonname.update(list(dial["scene_ids"].values()))

            image_embed(args, image_data, all_jsonname, data_type)
    else:
        all_jsonname = set()
        with open(os.path.join(args.data_folder, 'simmc2.1_dials_dstc11_teststd_public.json'), 'r') as f:
            dialogs = json.load(f)
            for dial in dialogs["dialogue_data"]:
                all_jsonname.update(list(dial["scene_ids"].values()))
        image_data = {p:os.path.join(args.scene_image_folder, p) for p in os.listdir(args.scene_image_folder)}

        image_embed(args, image_data, all_jsonname, "teststd")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", required=True, type=str, help="Path to SIMMC data file"
    )
    parser.add_argument(
        "--data_mode", 
        required=True, 
        choices=["public", "teststd"],
        type=str,
        help="Choice data mode [public, teststd]"
    )
    parser.add_argument(
        "--scene_image_folder", 
        required=True, 
        default=None, 
        type=str,
        help="Path to SIMMC scene images"
    )
    parser.add_argument(
        "--scene_json_folder", 
        required=True, 
        default=None, 
        type=str,
        help="Path to SIMMC scene jsons"
    )
    parser.add_argument(
        "--save_folder",
        required=True,
        type=str,
        help="Path to save",
    )

    args = parser.parse_args()

    main(args)