# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch

from collections import defaultdict
from transformers import BertTokenizer
from tqdm import tqdm

def main(args):
    tokenizer = BertTokenizer.from_pretrained('Luyu/co-condenser-wiki')
    if args.data_mode == "train":
        if args.simmc_train_json is None:
            raise ValueError('Check argument simmc_train_json')
        if args.metadata_folder is None:
            raise ValueError('Check argument metadata_folder')
        if args.scene_json_folder is None:
            raise ValueError('Check argument scene_json_folder')
        all_jsonname = set()
        with open(args.simmc_train_json, 'r') as f:
            dialogs = json.load(f)
            for dial in dialogs["dialogue_data"]:
                all_jsonname.update(list(dial["scene_ids"].values()))
        json_path = os.path.join(args.scene_json_folder)

        with open(os.path.join(args.metadata_folder, 'fashion_prefab_metadata_all.json'), 'r') as f:
            fashion_prefab = json.load(f)
        with open(os.path.join(args.metadata_folder, 'furniture_prefab_metadata_all.json'), 'r') as f:
            furniture_prefab = json.load(f)

        obj_text_inputs = defaultdict(dict)

        for scene_name in tqdm(all_jsonname, desc='Visual metadata preprocessing...'):
            with open(os.path.join(json_path, '{}_scene.json'.format(scene_name)), 'r', encoding='utf-8') as f:
                scene_json = json.load(f)
            for obj in scene_json['scenes'][0]['objects']:
                prefab_path = obj['prefab_path']
                if prefab_path in fashion_prefab:
                    if fashion_prefab[prefab_path]['sleeveLength'] == '':
                        sentences = 'This is {} with {} pattern and colored {}.'.format(fashion_prefab[prefab_path]['type'], fashion_prefab[prefab_path]['pattern'], fashion_prefab[prefab_path]['color'])
                    else:
                        sentences = 'This is {} sleeve {} with {} pattern and colored {}.'.format(fashion_prefab[prefab_path]['sleeveLength'], fashion_prefab[prefab_path]['type'], fashion_prefab[prefab_path]['pattern'], fashion_prefab[prefab_path]['color'])
                elif prefab_path in furniture_prefab:
                    if furniture_prefab[prefab_path]['type'] == 'CouchChair':
                        sentences = 'This is {} {}.'.format(furniture_prefab[prefab_path]['color'], 'Couch Chair')
                    elif furniture_prefab[prefab_path]['type'] == 'CoffeeTable':
                        sentences = 'This is {} {}.'.format(furniture_prefab[prefab_path]['color'], 'Coffee Table')
                    elif furniture_prefab[prefab_path]['type'] == 'EndTable':
                        sentences = 'This is {} {}.'.format(furniture_prefab[prefab_path]['color'], 'End Table')
                    elif furniture_prefab[prefab_path]['type'] == 'AreaRug':
                        sentences = 'This is {} {}.'.format(furniture_prefab[prefab_path]['color'], 'Area Rug')
                    else:
                        sentences = 'This is {} {}.'.format(furniture_prefab[prefab_path]['color'], furniture_prefab[prefab_path]['type'])
                else:
                    raise ValueError("Check prefab_path")

                d = tokenizer(sentences, add_special_tokens=True, padding='max_length', truncation=True, return_tensors="pt", max_length=20)
                obj_text_inputs[scene_name][obj['index']] = (d['input_ids'][0], d['attention_mask'][0], d['token_type_ids'][0])
        obj_text_inputs['no_answer'] = (torch.zeros(20, dtype=torch.long), torch.zeros(20, dtype=torch.long), torch.zeros(20, dtype=torch.long))
        torch.save(obj_text_inputs, os.path.join(args.save_folder, 'obj_text_feature_{}.bin'.format(args.data_mode)))
    else:
        if args.predicted_json_folder is None:
            raise ValueError('Check argument predicted_json_folder')
        for data_type in ["dev", "devtest", "teststd"]:
            if os.path.exists(os.path.join(args.predicted_json_folder, "fashion_{}_prediction.json".format(data_type))) and \
                os.path.exists(os.path.join(args.predicted_json_folder, "furniture_{}_prediction.json".format(data_type))):
                with open(os.path.join(args.predicted_json_folder, "fashion_{}_prediction.json".format(data_type)), 'r') as f:
                    fashion = json.load(f)
                with open(os.path.join(args.predicted_json_folder, "furniture_{}_prediction.json".format(data_type)), 'r') as f:
                    furniture = json.load(f)

                obj_text_inputs = defaultdict(dict)
                for scene_name, objects in fashion.items():
                    for index, obj in objects.items():
                        if obj['sleeveLength'] == '':
                            sentences = 'This is {} with {} pattern and colored {}.'.format(obj['type'], obj['pattern'], ', '.join(obj['color']))
                        else:
                            sentences = 'This is {} sleeve {} with {} pattern and colored {}.'.format(obj['sleeveLength'], obj['type'], obj['pattern'], ', '.join(obj['color']))

                        d = tokenizer(sentences, add_special_tokens=True, padding='max_length', truncation=True, return_tensors="pt", max_length=20)
                        obj_text_inputs[scene_name][int(index)] = (d['input_ids'][0], d['attention_mask'][0], d['token_type_ids'][0])
                    
                for scene_name, objects in furniture.items():
                    for index, obj in objects.items():
                        if obj['type'] == 'CouchChair':
                            sentences = 'This is {} {}.'.format(obj['color'], 'Couch Chair')
                        elif obj['type'] == 'CoffeeTable':
                            sentences = 'This is {} {}.'.format(obj['color'], 'Coffee Table')
                        elif obj['type'] == 'EndTable':
                            sentences = 'This is {} {}.'.format(obj['color'], 'End Table')
                        elif obj['type'] == 'AreaRug':
                            sentences = 'This is {} {}.'.format(obj['color'], 'Area Rug')
                        else:
                            sentences = 'This is {} {}.'.format(obj['color'], obj['type'])

                        d = tokenizer(sentences, add_special_tokens=True, padding='max_length', truncation=True, return_tensors="pt", max_length=20)
                        obj_text_inputs[scene_name][int(index)] = (d['input_ids'][0], d['attention_mask'][0], d['token_type_ids'][0])

                torch.save(obj_text_inputs, os.path.join(args.save_folder, 'obj_text_feature_{}.bin'.format(data_type)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_folder", default=None, help="Path to SIMMC 2.1 image metadata dir"
    )
    parser.add_argument(
        "--simmc_train_json", default=None, help="Path to SIMMC 2.1 train"
    )
    parser.add_argument(
        "--scene_json_folder", default=None, help="Path to SIMMC scene jsons"
    )
    parser.add_argument(
        "--predicted_json_folder", default=None, help="Path to predicted metadata json"
    )
    parser.add_argument(
        "--data_mode", 
        required=True, 
        choices=["train", "predicted_data"],
        type=str,
        help="Choice data mode [train, predicted_data]"
    )
    parser.add_argument(
        "--save_folder",
        required=True,
        type=str,
        help="Path to save",
    )

    args = parser.parse_args()

    main(args)