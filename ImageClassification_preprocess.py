# -*- coding: utf-8 -*-

import argparse
import json
import os

from collections import defaultdict
from tqdm import tqdm

error_image = ["cloth_store_1416238_woman_19_0", "cloth_store_1416238_woman_4_8", "cloth_store_1416238_woman_20_6", "m_cloth_store_1416238_woman_19_0", "m_cloth_store_1416238_woman_4_8", "m_cloth_store_1416238_woman_20_6"]

def main(args):
    with open(os.path.join(args.data_folder, 'fashion_prefab_metadata_all.json'), 'r') as f:
        fashion_prefab = json.load(f)
    with open(os.path.join(args.data_folder, 'furniture_prefab_metadata_all.json'), 'r') as f:
        furniture_prefab = json.load(f)
        
    if args.data_mode == "public":
        all_jsonname = defaultdict(set)
        for datatype in ['train', 'dev', 'devtest']:
            with open(os.path.join(args.data_folder, 'simmc2.1_dials_dstc11_{}.json'.format(datatype)), 'r') as f:
                dialogs = json.load(f)
                for dial in dialogs["dialogue_data"]:
                    all_jsonname[datatype].update(list(dial["scene_ids"].values()))

        json_path = os.path.join(args.scene_json_folder)

        fashion_prefab_data = defaultdict(set)
        furniture_prefab_data = defaultdict(set)
        for datatype in all_jsonname:
            for scene_name in tqdm(all_jsonname[datatype], desc='image preprocessing...'):
                if scene_name in error_image:
                    continue

                with open(os.path.join(json_path, '{}_scene.json'.format(scene_name)), 'r', encoding='utf-8') as f:
                    scene_json = json.load(f)
                image_name = scene_name.replace('m_', '') + ".png"
                for obj in scene_json['scenes'][0]['objects']:
                    if 'cloth' in image_name:
                        fashion_prefab_data[datatype].add('\t'.join([scene_name, image_name, obj['prefab_path'], ' '.join(map(str, obj['bbox'])), str(obj['index'])]))
                    else:
                        furniture_prefab_data[datatype].add('\t'.join([scene_name, image_name, obj['prefab_path'], ' '.join(map(str, obj['bbox'])), str(obj['index'])]))
        
        # split data
        for datatype in fashion_prefab_data:
            print('fashion {} # instance : {}'.format(datatype, str(len(fashion_prefab_data[datatype]))))
        
        # no have dev furniture
        furniture_prefab_data['dev'] = set(list(furniture_prefab_data['devtest'])[:20])

        for datatype in furniture_prefab_data:
            print('furniture {} # instance : {}'.format(datatype, str(len(furniture_prefab_data[datatype]))))

        fashion = defaultdict(set)
        for data in (fashion_prefab_data['train'] | fashion_prefab_data['dev'] | fashion_prefab_data['devtest']):
            prefab = data.split('\t')[2]
            d = fashion_prefab[prefab]
            for t in ['color', 'type', 'sleeveLength', 'pattern']:
                if t == 'color':
                    fashion[t].update(d[t].split(', '))
                else:
                    fashion[t].add(d[t])
        for t in ['color', 'type', 'sleeveLength', 'pattern']:
            with open(os.path.join(args.save_folder, 'fashion_{}.json'.format(t)), 'w', encoding='utf-8') as f:
                data = {_:idx for idx, _ in enumerate(sorted(fashion[t]))}
                json.dump(data, f, indent='\t')

        furniture = defaultdict(set)
        for data in (furniture_prefab_data['train'] | furniture_prefab_data['dev'] | furniture_prefab_data['devtest']):
            prefab = data.split('\t')[2]
            d = furniture_prefab[prefab]
            for t in ['color', 'type']:
                furniture[t].add(d[t])
        for t in ['color', 'type']:
            with open(os.path.join(args.save_folder, 'furniture_{}.json'.format(t)), 'w', encoding='utf-8') as f:
                data = {_:idx for idx, _ in enumerate(sorted(furniture[t]))}
                json.dump(data, f, indent='\t')

        color = defaultdict(int)
        t_length = 0
        for domain, data in [('fashion', fashion_prefab_data), ('furniture', furniture_prefab_data)]:
            for datatype, value in data.items():
                data_list = list()
                inference_data_list = list()
                for d in value:
                    tmp = dict()
                    inf_tmp = dict()
                    scene_name, image_name, prefab, bbox, index = d.split('\t')
                    tmp['image_name'] = image_name
                    tmp['scene_name'] = scene_name
                    inf_tmp['image_name'] = image_name
                    inf_tmp['scene_name'] = scene_name
                    tmp['prefab'] = prefab
                    tmp['bbox'] = list(map(int, bbox.split(' ')))
                    inf_tmp['bbox'] = list(map(int, bbox.split(' ')))
                    tmp['index'] = index
                    inf_tmp['index'] = index
                    if domain == 'fashion':
                        for t in ['color', 'type', 'sleeveLength', 'pattern']:
                            inf_tmp[t] = None
                            if t == 'color':
                                tmp[t] = fashion_prefab[prefab][t].split(', ')
                                for c in fashion_prefab[prefab][t].split(', '):
                                    color[c] += 1
                                t_length += 1
                            else:
                                tmp[t] = fashion_prefab[prefab][t]
                    else:
                        for t in ['color', 'type']:
                            inf_tmp[t] = None
                            tmp[t] = furniture_prefab[prefab][t]

                    data_list.append(tmp)
                    inference_data_list.append(inf_tmp)
                if datatype == 'train':
                    new_color = dict()
                    for k in sorted(color.keys()):
                        new_color[k] = (t_length - color[k]) // color[k]
                    with open(os.path.join(args.save_folder, "fashion_color_weight.json"), 'w', encoding='utf-8') as f:
                        json.dump(new_color, f, indent='\t')
                else:
                    if datatype == 'dev' and domain == 'furniture':
                        with open(os.path.join(args.save_folder, '{}_{}_testing.json'.format(domain, datatype)), 'w', encoding='utf-8') as f:
                            json.dump([], f, indent='\t')
                    with open(os.path.join(args.save_folder, '{}_{}_testing.json'.format(domain, datatype)), 'w', encoding='utf-8') as f:
                        json.dump(inference_data_list, f, indent='\t')
                with open(os.path.join(args.save_folder, '{}_{}.json'.format(domain, datatype)), 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent='\t')
    else:
        all_jsonname = set()
        with open(os.path.join(args.data_folder, 'simmc2.1_dials_dstc11_teststd_public.json'), 'r') as f:
            dialogs = json.load(f)
            for dial in dialogs["dialogue_data"]:
                all_jsonname.update(list(dial["scene_ids"].values()))
            
        json_path = os.path.join(args.scene_json_folder)
        
        fashion_prefab_data = set()
        furniture_prefab_data = set()
        for scene_name in tqdm(all_jsonname, desc='image preprocessing...'):
            if scene_name in error_image:
                continue

            with open(os.path.join(json_path, '{}_scene.json'.format(scene_name)), 'r', encoding='utf-8') as f:
                scene_json = json.load(f)

            image_name = scene_name.replace('m_', '') + ".png"
            for obj in scene_json['scenes'][0]['objects']:
                if 'cloth' in image_name:
                    fashion_prefab_data.add('\t'.join([scene_name, image_name, ' '.join(map(str, obj['bbox'])), str(obj['index'])]))
                else:
                    furniture_prefab_data.add('\t'.join([scene_name, image_name, ' '.join(map(str, obj['bbox'])), str(obj['index'])]))
                    
        for domain, data in [('fashion', fashion_prefab_data), ('furniture', furniture_prefab_data)]:
            data_list = list()
            for d in data:
                tmp = dict()
                scene_name, image_name,  bbox, index = d.split('\t')
                tmp['image_name'] = image_name
                tmp['scene_name'] = scene_name
                tmp['bbox'] = list(map(int, bbox.split(' ')))
                tmp['index'] = index
                if domain == 'fashion':
                    for t in ['color', 'type', 'sleeveLength', 'pattern']:
                        tmp[t] = None
                else:
                    for t in ['color', 'type']:
                        tmp[t] = None

                data_list.append(tmp)
            with open(os.path.join(args.save_folder, '{}_teststd.json'.format(domain)), 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", required=True, type=str, help="Path to SIMMC data file"
    )
    parser.add_argument(
        "--scene_json_folder", required=True, default=None, help="Path to SIMMC scene jsons"
    )
    parser.add_argument(
        "--data_mode", 
        required=True, 
        choices=["public", "teststd"],
        type=str,
        help="Choice data mode [public, teststd]"
    )
    parser.add_argument(
        "--save_folder",
        required=True,
        type=str,
        help="Path to save",
    )

    args = parser.parse_args()

    main(args)