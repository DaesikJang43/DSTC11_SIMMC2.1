# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from transformers import BertConfig, BertModel, ViTConfig

class Disambiguation_Detection(nn.Module):
    def __init__(self, args):
        super(Disambiguation_Detection, self).__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model, config=self.bert_config)

        self.vit_config = ViTConfig.from_pretrained(args.vit_model)

        self.bbox_layer = nn.Linear(self.vit_config.hidden_size + 4, self.vit_config.hidden_size, bias=False)
        
        # concat and integration
        self.integration_layer = nn.Linear(self.vit_config.hidden_size * 2 , self.vit_config.hidden_size, bias=False)
        self.obj_integration_layer = nn.Linear(self.vit_config.hidden_size + self.bert_config.hidden_size, self.vit_config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        scene_embed: Optional[torch.Tensor] = None,
        obj_embed: Optional[torch.Tensor] = None,
        obj_bbox: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        BCE_labels: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
        obj_input_ids: Optional[torch.Tensor] = None,
        obj_attention_mask: Optional[torch.Tensor] = None,
        obj_token_type_ids: Optional[torch.Tensor] = None,
    ):

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[1]

        if task == "1":
            batch_size, num_samples, max_length = obj_input_ids.shape
            obj_text_embed = torch.zeros((batch_size * num_samples, self.bert_config.hidden_size), dtype=torch.float, device=obj_input_ids.device)
            obj_input_ids = obj_input_ids.view(-1, max_length)
            obj_attention_mask = obj_attention_mask.view(-1, max_length)
            obj_token_type_ids = obj_token_type_ids.view(-1, max_length)
            nonzero_feature_idx = torch.nonzero(torch.sum(obj_input_ids != 0, dim=-1) != 0, as_tuple=True)
            obj_text_embed[nonzero_feature_idx] = self.bert(
                obj_input_ids[nonzero_feature_idx],
                attention_mask=obj_attention_mask[nonzero_feature_idx],
                token_type_ids=obj_token_type_ids[nonzero_feature_idx]
            )[1]
            obj_text_embed = obj_text_embed.view(batch_size, num_samples, -1)
        else:
            obj_text_embed = self.bert(
                obj_input_ids,
                attention_mask=obj_attention_mask,
                token_type_ids=obj_token_type_ids
            )[1]

        scene_embed = torch.cat((scene_embed, obj_bbox), dim=-1)
        scene_embed = self.bbox_layer(scene_embed)
        image_embed = torch.cat((scene_embed, obj_embed), dim=-1)

        image_rate = F.sigmoid(self.integration_layer(image_embed))
        image_output = image_rate * scene_embed + (1 - image_rate) * obj_embed

        obj_image_text_embed = torch.cat((image_output, obj_text_embed), dim=-1)
        obj_rate = F.sigmoid(self.obj_integration_layer(obj_image_text_embed))
        obj_output = obj_rate * image_output + (1 - obj_rate) * obj_text_embed

        if task == "1":
            logits = torch.matmul(bert_output.unsqueeze(1), obj_output.transpose(-1, -2)).squeeze(1)
        else:
            logits = (bert_output * obj_output).sum(-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.div(self.args.temperature).view(input_ids.shape[0], -1), labels.view(-1))
        if BCE_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.args.positive_weight], device=BCE_labels.device))
            loss = loss_fct(logits.view(-1, 1), BCE_labels.view(-1, 1))

        return logits if labels is None and BCE_labels is None else (logits, loss)