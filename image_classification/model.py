# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from transformers import ViTConfig, ViTModel

class ImageClassifier(nn.Module):
    def __init__(self, args):
        super(ImageClassifier, self).__init__()
        self.args = args

        self.vit_config = ViTConfig.from_pretrained(args.vit_model)
        self.vit = ViTModel.from_pretrained(args.vit_model, config=self.vit_config)

        if "1" in args.task:
            self.type_layer = nn.Linear(self.vit_config.hidden_size, args.num_labels['type'])
        if "2" in args.task:
            self.color_layer = nn.Linear(self.vit_config.hidden_size, args.num_labels['color'])
        if self.args.domain == 'fashion':
            if "3" in args.task:
                self.pattern_layer = nn.Linear(self.vit_config.hidden_size, args.num_labels['pattern'])
            if "4" in args.task:
                self.sleeve_layer = nn.Linear(self.vit_config.hidden_size, args.num_labels['sleeveLength'])

    def forward(
        self,
        pixel_values: torch.Tensor = None,
        type_labels: Optional[torch.Tensor] = None,
        color_labels: Optional[torch.Tensor] = None,
        pattern_labels: Optional[torch.Tensor] = None,
        sleeve_labels: Optional[torch.Tensor] = None,
    ):
        image_embed = self.vit(pixel_values)[1]

        logits = tuple()
        if "1" in self.args.task:
            type_logits = self.type_layer(image_embed)
            logits = logits + (type_logits,)
        else:
            logits = logits + (None,)
        if "2" in self.args.task:
            color_logits = self.color_layer(image_embed)
            logits = logits + (color_logits,)
        else:
            logits = logits + (None,)

        if self.args.domain =='fashion':
            if "3" in self.args.task:
                pattern_logits = self.pattern_layer(image_embed)
                logits = logits + (pattern_logits,)
            else:
                logits = logits + (None,)
            if "4" in self.args.task:
                sleeve_logits = self.sleeve_layer(image_embed)
                logits = logits + (sleeve_logits,)
            else:
                logits = logits + (None,)

        if "1" in self.args.task and type_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            type_loss = loss_fct(type_logits, type_labels.view(-1))
        if "2" in self.args.task and color_labels is not None:
            if self.args.domain =='fashion':
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.args.positive_weight], device=color_labels.device))
                color_loss = loss_fct(color_logits, color_labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                color_loss = loss_fct(color_logits, color_labels.view(-1))
        if "3" in self.args.task and pattern_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            pattern_loss = loss_fct(pattern_logits, pattern_labels.view(-1))
        if "4" in self.args.task and sleeve_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sleeve_loss = loss_fct(sleeve_logits, sleeve_labels.view(-1))

        loss = None
        if len(self.args.task) == 4 and self.args.domain =='fashion':
            if type_labels is not None:
                loss = type_loss + color_loss + pattern_loss + sleeve_loss
        elif len(self.args.task) >= 2 and self.args.domain !='fashion':
            if type_labels is not None:
                loss = type_loss + color_loss
        elif "1" in self.args.task and type_labels is not None:
            loss = type_loss
        elif "2" in self.args.task and color_labels is not None:
            loss = color_loss
        elif "3" in self.args.task and pattern_labels is not None:
            loss = pattern_loss
        elif "4" in self.args.task and sleeve_labels is not None:
            loss = sleeve_loss
        else:
            raise ValueError("Check Your task!")

        return logits if type_labels is None else (logits, loss)
