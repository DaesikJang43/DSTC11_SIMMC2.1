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

@dataclass(frozen=True)
class simmc_data:
    dialog_id: int
    turn_id: int
    input_text: List[str]
    label: Optional[int]

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

def load_data(data_path, eval=False):
    samples = list()
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data['data']:
            if not eval:
                samples.append(
                    simmc_data(
                        dialog_id = d["dialog_id"],
                        turn_id = d["turn_id"],
                        input_text = d["input_text"],
                        label = d["ambiguous_label"]
                    )
                )
            else:
                samples.append(
                    simmc_data(
                        dialog_id = d["dialog_id"],
                        turn_id = d["turn_id"],
                        input_text = d["input_text"],
                        label = None
                    )
                )

    return samples

def convert_samples_to_features(args, samples, tokenizer):
    max_utterances = 2 * args.max_turns + 1
    history_text = list()
    last_text = list()
    labels = list()
    for sample in tqdm(samples, desc="Converting..."):
        dialog = sample.input_text
        history = list()
        for utt_id, utt in enumerate(dialog[-max_utterances:]):
            if utt_id % 2 == 0:
                history.append("<USER> " + utt)
            else:
                history.append("<SYS> " + utt)
        last_text.append(history[-1])
        history_text.append(" ".join(history[:-1]))
        
        if sample.label is not None:
            labels.append(sample.label)

    features = tokenizer(last_text, history_text, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True)
    if sample.label is not None:
        features['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return TensorDataset(features['input_ids'], features['attention_mask'], features['token_type_ids'], features['labels'])
        
    return TensorDataset(features['input_ids'], features['attention_mask'], features['token_type_ids'])
