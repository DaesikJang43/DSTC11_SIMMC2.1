# -*- coding: utf-8 -*-

import re
import json
import logging
import numpy as np
import os
import random
import torch

from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm, trange
from typing import List, Optional
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

error_image = ["cloth_store_1416238_woman_19_0", "cloth_store_1416238_woman_4_8", "cloth_store_1416238_woman_20_6", "m_cloth_store_1416238_woman_19_0", "m_cloth_store_1416238_woman_4_8", "m_cloth_store_1416238_woman_20_6"]

@dataclass(frozen=True)
class simmc_data:
    dialog_id: int
    turn_id: int
    input_text: List[str]
    scene_ids: Optional[str]
    candidates: Optional[List[int]]
    object_map: List[int]

def order_selection(tasks, number_of_batches):
    order = []
    picked_num = {task: 0 for task in tasks}

    while(all(number_of_batches[task] == 0 for task in number_of_batches) is not True):
        total = sum([number_of_batches[tasks] for tasks in number_of_batches])
        prob = [number_of_batches[task]/total for task in number_of_batches]
        pick = np.random.choice(tasks, p=prob)

        order.append((pick, picked_num[pick]))
        number_of_batches[pick] -= 1
        picked_num[pick] += 1

    return order

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
        for d in data['data']:
            if d["ambiguous_label"] == 1 and d["scene_ids"] not in error_image:
                samples.append(
                    simmc_data(
                        dialog_id = d["dialog_id"],
                        turn_id = d["turn_id"],
                        input_text = d["input_text"],
                        scene_ids = d["scene_ids"],
                        candidates = d["ambiguous_candidates"],
                        object_map = d["object_map"]
                    )
                )

    return samples

def convert_samples_to_features(args, samples, tokenizer, eval=False, task="1"):
    max_utterances = 2 * args.max_turns + 1
    history_text = list()
    last_text = list()
    extra_features = defaultdict(list)
    labels = list()
    scenes = list()
    for sample in tqdm(samples, desc="Converting..."):
        dialog = sample.input_text
        history = list()
        for utt_id, utt in enumerate(dialog[-max_utterances:]):
            if utt_id % 2 == 0:
                history.append("<USER> " + utt)
            else:
                history.append("<SYS> " + utt)
        if task == "1":
            if sample.candidates is None or eval:
                last_text.append(history[-1])
                history_text.append(" ".join(history[:-1]))
                extra_features['scenes'].append(sample.scene_ids)
                extra_features['dialog_id'].append(sample.dialog_id)
                extra_features['turn_id'].append(sample.turn_id)
                extra_features['object_map'].append(sample.object_map)
                if eval:
                    extra_features['labels'].append(sample.candidates)
            elif len(sample.candidates) == 0:
                last_text.append(history[-1])
                history_text.append(" ".join(history[:-1]))
                extra_features['labels'].append(('no_answer', sample.candidates))
                extra_features['scenes'].append(sample.scene_ids)
            elif len(sample.candidates) > 0:
                for cand in sample.candidates:
                    last_text.append(history[-1])
                    history_text.append(" ".join(history[:-1]))
                    extra_features['labels'].append((cand, sample.candidates))
                    extra_features['scenes'].append(sample.scene_ids)
            else:
                raise ValueError("Util file Error")
        else:
            if len(sample.candidates) == 0:
                for cand in sample.object_map:
                    last_text.append(history[-1])
                    history_text.append(" ".join(history[:-1]))
                    extra_features['scenes'].append(sample.scene_ids)
                    extra_features['labels'].append((0.0, cand))
            elif len(sample.candidates) > 0:
                for cand in sample.object_map:
                    last_text.append(history[-1])
                    history_text.append(" ".join(history[:-1]))
                    extra_features['scenes'].append(sample.scene_ids)
                    if cand in sample.candidates:
                        extra_features['labels'].append((1.0, cand))
                    else:
                        extra_features['labels'].append((0.0, cand))
            else:
                raise ValueError("Util file Error")
                
    features = tokenizer(last_text, history_text, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True)

    for key, value in extra_features.items():
        features[key] = value
        
    return features