import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

import models.open_clip.src.open_clip as oc

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIP(nn.Module):
    def __init__(self, args, map_location='cpu', device='cpu', clip_model_name='laclip', replicate_mmae=False):
        super(MV_CLIP, self).__init__()
        
        self.model = oc.factory.create_model(model_name='ViT-B-32', precision='amp', force_quick_gelu=True)
        if clip_model_name=='laclip':
          chkt = torch.load('/content/drive/MyDrive/MMSD_project/laion400m_laclip.pt', map_location=map_location)
        elif clip_model_name=='clip':
          chkt = torch.load('/content/drive/MyDrive/MMSD_project/laion400m_clip.pt', map_location=map_location)
        else:
          raise ValueError('Not a valid model type')
        self.model.load_state_dict(chkt['state_dict'], strict=True)
        self.model.to(device)

        self.replicate_mmae = replicate_mmae
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        if self.replicate_mmae:
            self.config.hidden_size = 768
            self.config.num_attention_heads = 12
        else:
            self.config.hidden_size = 512
            self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        self.trans.to(device)
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        self.ln_img_embed = nn.LayerNorm(args.image_size)
        self.ln_text_embed = nn.LayerNorm(args.text_size)
        if self.replicate_mmae:
            self.text_projection = nn.Linear(args.text_size, args.image_size, bias=False)
            self.image_projection = nn.Linear(args.image_size, args.image_size, bias=False)
            self.classifier_fuse = nn.Linear(args.image_size , args.label_number)
            self.classifier_text = nn.Linear(args.text_size, args.label_number)
            self.classifier_image = nn.Linear(args.image_size, args.label_number)

            self.loss_fct = nn.CrossEntropyLoss()
            self.att = nn.Linear(args.image_size, 1, bias=False)
        else:
            self.text_projection = nn.Linear(args.text_size, args.text_size, bias=False)
            self.image_projection = nn.Linear(args.image_size, args.text_size, bias=False)
            self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
            self.classifier_text = nn.Linear(args.text_size, args.label_number)
            self.classifier_image = nn.Linear(args.image_size, args.label_number)

            self.loss_fct = nn.CrossEntropyLoss()
            self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, image, text, padding_mask, input_ids, labels):
        output = self.model(image, text, padding_mask) 
        text_features = output['text_features']
        image_features = output['image_features']
        text_feature = output['text_feature']
        image_feature = output['image_feature']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        text_embeds = self.text_projection(self.ln_text_embed(text_features))
        image_embeds = self.image_projection(self.ln_img_embed(image_features))
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), 1.0 - padding_mask), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_feature = fuse_hiddens[:, 50:, :].mean(axis=1)
        # new_text_feature = new_text_features[
        #     torch.arange(new_text_features.shape[0], device=input_ids.device), input_ids.to(torch.int).argmax(dim=-1)
        # ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image

            outputs = (loss,) + outputs
        return outputs


