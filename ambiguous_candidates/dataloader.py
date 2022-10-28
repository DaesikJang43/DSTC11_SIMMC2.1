# -*- coding: utf-8 -*-

import os
import random
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

class TrainDataset_1(Dataset):
    def __init__(self, args, features, obj_embed, scene_embed, bbox_embed, obj_text):
        self.len = len(features['input_ids'])
        self.features = features
        self.negative_sample_size = args.negative_sample_size
        self.obj_embed = obj_embed
        self.scene_embed = scene_embed
        self.bbox_embed = bbox_embed
        self.obj_text = obj_text
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        input_ids = self.features['input_ids'][idx]
        attention_mask = self.features['attention_mask'][idx]
        token_type_ids = self.features['token_type_ids'][idx]
        
        if self.features['labels'][idx][0] == 'no_answer':
            scene_embedding = [self.obj_embed['no_answer']]
        else:
            scene_embedding = [self.scene_embed[self.features['scenes'][idx]]]

        if self.features['labels'][idx][0] == 'no_answer':
            pos_obj_embed = [self.obj_embed['no_answer']]
            pos_obj_bbox = [self.bbox_embed['no_answer']]
            pos_obj_text = [self.obj_text['no_answer']]
            neg_pool = [(v, self.bbox_embed[self.features['scenes'][idx]][k], self.obj_text[self.features['scenes'][idx]][k]) for k, v in self.obj_embed[self.features['scenes'][idx]].items() if k not in self.features['labels'][idx][1]]
        else:
            pos_obj_embed = [self.obj_embed[self.features['scenes'][idx]][self.features['labels'][idx][0]]]
            pos_obj_bbox = [self.bbox_embed[self.features['scenes'][idx]][self.features['labels'][idx][0]]]
            pos_obj_text = [self.obj_text[self.features['scenes'][idx]][self.features['labels'][idx][0]]]
            neg_pool = [(v, self.bbox_embed[self.features['scenes'][idx]][k], self.obj_text[self.features['scenes'][idx]][k]) for k, v in self.obj_embed[self.features['scenes'][idx]].items() if k not in self.features['labels'][idx][1]] + [(self.obj_embed['no_answer'], self.bbox_embed['no_answer'], self.obj_text['no_answer'])]
    
        if len(neg_pool) >= self.negative_sample_size:
            neg = random.sample(neg_pool, self.negative_sample_size)
        else:
            neg = random.choices(neg_pool, k=self.negative_sample_size)

        neg_obj_embed = list()
        neg_obj_bbox = list()
        neg_obj_text = list()
        for n in neg:
            neg_obj_embed.append(n[0])
            neg_obj_bbox.append(n[1])
            neg_obj_text.append(n[2])

        for neg in neg_obj_embed:
            if neg.sum() == 0.0:
                scene_embedding += [self.obj_embed['no_answer']]
            else:
                scene_embedding += [self.scene_embed[self.features['scenes'][idx]]]

        obj_input_ids = list()
        obj_attention_mask = list()
        obj_token_type_ids = list()
        for i1, i2, i3 in (pos_obj_text + neg_obj_text):
            obj_input_ids.append(i1)
            obj_attention_mask.append(i2)
            obj_token_type_ids.append(i3)
            
        obj_embedding = torch.stack(pos_obj_embed+neg_obj_embed, dim=0)
        obj_bbox = torch.stack(pos_obj_bbox + neg_obj_bbox, dim=0)
        obj_input_ids = torch.stack(obj_input_ids, dim=0)
        obj_attention_mask = torch.stack(obj_attention_mask, dim=0)
        obj_token_type_ids = torch.stack(obj_token_type_ids, dim=0)
        scene_embedding = torch.stack(scene_embedding, dim=0)

        labels = torch.tensor(0, dtype=torch.long) 
    
        return input_ids, attention_mask, token_type_ids, scene_embedding, obj_embedding, labels, \
                obj_bbox, obj_input_ids, obj_attention_mask, obj_token_type_ids

    @staticmethod
    def collate_fn(data):
        input_ids = torch.stack([_[0] for _ in data], dim=0)
        attention_mask = torch.stack([_[1] for _ in data], dim=0)
        token_type_ids = torch.stack([_[2] for _ in data], dim=0)

        scene_embedding = torch.stack([_[3] for _ in data], dim=0)
        obj_embedding = torch.stack([_[4] for _ in data], dim=0)
        labels = torch.stack([_[5] for _ in data])

        obj_bbox = torch.stack([_[6] for _ in data], dim=0)
        obj_input_ids = torch.stack([_[7] for _ in data], dim=0)
        obj_attention_mask = torch.stack([_[8] for _ in data], dim=0)
        obj_token_type_ids = torch.stack([_[9] for _ in data], dim=0)
            
        return input_ids, attention_mask, token_type_ids, scene_embedding, obj_embedding, labels, \
                obj_bbox, obj_input_ids, obj_attention_mask, obj_token_type_ids

class TrainDataset_2(Dataset):
    def __init__(self, args, features, obj_embed, scene_embed, bbox_embed, obj_text):
        self.len = len(features['input_ids'])
        self.features = features
        self.negative_sample_size = args.negative_sample_size
        self.obj_embed = obj_embed
        self.scene_embed = scene_embed
        self.bbox_embed = bbox_embed 
        self.obj_text = obj_text
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        input_ids = self.features['input_ids'][idx]
        attention_mask = self.features['attention_mask'][idx]
        token_type_ids = self.features['token_type_ids'][idx]

        obj_embedding = self.obj_embed[self.features['scenes'][idx]][self.features['labels'][idx][1]]
        obj_input_ids, obj_attention_mask, obj_token_type_ids = self.obj_text[self.features['scenes'][idx]][self.features['labels'][idx][1]]
        obj_bbox = self.bbox_embed[self.features['scenes'][idx]][self.features['labels'][idx][1]]
        scene_embedding = self.scene_embed[self.features['scenes'][idx]]

        labels = torch.tensor(self.features['labels'][idx][0], dtype=torch.float)
    
        return input_ids, attention_mask, token_type_ids, scene_embedding, obj_embedding, labels, \
                obj_bbox, obj_input_ids, obj_attention_mask, obj_token_type_ids

    @staticmethod
    def collate_fn(data):
        input_ids = torch.stack([_[0] for _ in data], dim=0)
        attention_mask = torch.stack([_[1] for _ in data], dim=0)
        token_type_ids = torch.stack([_[2] for _ in data], dim=0)

        scene_embedding = torch.stack([_[3] for _ in data], dim=0)
        obj_embedding = torch.stack([_[4] for _ in data], dim=0)
        labels = torch.stack([_[5] for _ in data])

        obj_bbox = torch.stack([_[6] for _ in data], dim=0)
        obj_input_ids = torch.stack([_[7] for _ in data], dim=0)
        obj_attention_mask = torch.stack([_[8] for _ in data], dim=0)
        obj_token_type_ids = torch.stack([_[9] for _ in data], dim=0)
            
        return input_ids, attention_mask, token_type_ids, scene_embedding, obj_embedding, labels, \
                obj_bbox, obj_input_ids, obj_attention_mask, obj_token_type_ids

class TestDataset(Dataset):
    def __init__(self, args, features, data_type):
        self.len = len(features['input_ids'])
        self.features = features
        self.data_type = data_type
        if os.path.exists(os.path.join(args.data_dir, "image_features", "obj_image_{}.bin".format(data_type))):
            self.obj_embed = torch.load(os.path.join(args.data_dir, "image_features", "obj_image_{}.bin".format(data_type)))
        else:
            raise ValueError('Check your obj_image_{}.bin file'.format(data_type))
        if os.path.exists(os.path.join(args.data_dir, "image_features", "scene_image_{}.bin".format(data_type))):
            self.scene_embed = torch.load(os.path.join(args.data_dir, "image_features", "scene_image_{}.bin".format(data_type)))
        else:
            raise ValueError('Check your scene_image_{}.bin file'.format(data_type))
        if os.path.exists(os.path.join(args.data_dir, "image_features", "obj_image_{}_normalized_bbox.bin".format(data_type))):
            self.bbox_embed = torch.load(os.path.join(args.data_dir, "image_features", "obj_image_{}_normalized_bbox.bin".format(data_type)))
        else:
            raise ValueError('Check your obj_image_{}_normalized_bbox.bin file'.format(data_type))
        if os.path.exists(os.path.join(args.data_dir, "image_features", "obj_text_feature_{}.bin".format(data_type))):
            self.obj_text = torch.load(os.path.join(args.data_dir, "image_features", "obj_text_feature_{}.bin".format(data_type)))
        else:
            raise ValueError('Check your obj_text_feature_{}.bin file'.format(data_type))

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        input_ids = self.features['input_ids'][idx]
        attention_mask = self.features['attention_mask'][idx]
        token_type_ids = self.features['token_type_ids'][idx]

        labels = list()
        scene_embedding = list()
        obj_embedding = list()
        obj_bbox = list()
        obj_input_ids = list()
        obj_attention_mask = list()
        obj_token_type_ids = list()

        for obj_idx in self.features['object_map'][idx]:
            obj_embedding.append(self.obj_embed[self.features['scenes'][idx]][obj_idx])
            obj_bbox.append(self.bbox_embed[self.features['scenes'][idx]][obj_idx])
            obj_input_ids.append(self.obj_text[self.features['scenes'][idx]][obj_idx][0])
            obj_attention_mask.append(self.obj_text[self.features['scenes'][idx]][obj_idx][1])
            obj_token_type_ids.append(self.obj_text[self.features['scenes'][idx]][obj_idx][2])

            scene_embedding.append(self.scene_embed[self.features['scenes'][idx]])

        obj_embedding = torch.stack(obj_embedding, dim=0)
        scene_embedding = torch.stack(scene_embedding, dim=0)
        obj_bbox = torch.stack(obj_bbox, dim=0)

        dialog_id = torch.tensor(self.features['dialog_id'][idx], dtype=torch.long)
        turn_id = torch.tensor(self.features['turn_id'][idx], dtype=torch.long)
        object_map = torch.tensor(self.features['object_map'][idx], dtype=torch.long)

        obj_input_ids = torch.stack(obj_input_ids, dim=0)
        obj_attention_mask = torch.stack(obj_attention_mask, dim=0)
        obj_token_type_ids = torch.stack(obj_token_type_ids, dim=0)
            
        return input_ids, attention_mask, token_type_ids, scene_embedding, obj_embedding, \
            dialog_id, turn_id, object_map, obj_bbox, obj_input_ids, obj_attention_mask, obj_token_type_ids

    @staticmethod
    def collate_fn(data):
        input_ids = torch.stack([_[0] for _ in data], dim=0)
        attention_mask = torch.stack([_[1] for _ in data], dim=0)
        token_type_ids = torch.stack([_[2] for _ in data], dim=0)

        scene_embedding = torch.stack([_[3] for _ in data], dim=0)
        obj_embedding = torch.stack([_[4] for _ in data], dim=0)
        dialog_id = torch.stack([_[5] for _  in data], dim=0)
        turn_id = torch.stack([_[6] for _  in data], dim=0)

        object_map = torch.stack([_[7] for _ in data], dim=0)
        obj_bbox = torch.stack([_[8] for _ in data], dim=0)
        obj_input_ids = torch.stack([_[9] for _ in data], dim=0)
        obj_attention_mask = torch.stack([_[10] for _ in data], dim=0)
        obj_token_type_ids = torch.stack([_[11] for _ in data], dim=0)
            
        return input_ids, attention_mask, token_type_ids, scene_embedding, obj_embedding, \
                dialog_id, turn_id, object_map, obj_bbox, obj_input_ids, obj_attention_mask, obj_token_type_ids