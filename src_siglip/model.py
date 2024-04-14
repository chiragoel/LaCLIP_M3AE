from transformers import CLIPModel,BertConfig,AutoModel
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import laclip_model_arch
import numpy as np

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        # print("BERT layer config : ", self.layer , "\n----------------")
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            # print("BERT hid state attention : ", hidden_states.shape, attention.shape)            
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIP(nn.Module):
    def __init__(self, args):
        super(MV_CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
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

        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True) #CLIPOutput
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

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


class LACLIP(nn.Module):
    def __init__(self, args):
        super(LACLIP, self).__init__()

        laclip_path = "/home/zeus/MMSD2.0/pretrained_models/laion400m_laclip.pt"
        laion_clip_path = "/home/zeus/MMSD2.0/pretrained_models/laion400m_clip.pt"
        self.model = getattr(laclip_model_arch, "CLIP_VITB32")()
        # self.model.load_state_dict(torch.load(laclip_path), strict=False)
        self.model.load_state_dict(torch.load(laion_clip_path), strict=False)
        

        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
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

        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels):
        output = self.model(inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask'])  #output_attention called but not used in MV_CLIP
        # text_features = output['text_model_output']['last_hidden_state']
        # image_features = output['vision_model_output']['last_hidden_state']
        text_features = output['text_last_hidden_state']
        image_features = output['image_last_hidden_state']
        # text_feature = output['text_model_output']['pooler_output']
        # image_feature = output['vision_model_output']['pooler_output']
        text_feature = output['text_pooled_output']
        image_feature = output['image_pooled_output']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)
        # text_embeds = self.model.text_projection(text_features)
        # image_embeds = self.model.visual_projection(image_features)
        text_embeds = output['text_proj_last_hidden_state']
        image_embeds = output['image_proj_last_hidden_state']

        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

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



class SIGLIP(nn.Module):
    def __init__(self, args):
        super(SIGLIP, self).__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        # print("self.model ----", self.model)
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        if args.simple_linear:
            self.text_linear =  nn.Linear(768, args.text_size)  #downsample from 768 to 512
            self.image_linear =  nn.Linear(args.image_size, 512) #downsample from 768 to 512
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(768, args.text_size), #downsample from 768 to 512
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, 512), #downsample from 768 to 512
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        #adding text and image projections to Siglip
        self.text_projection = nn.Parameter(torch.empty(768, 512))
        self.image_projection = nn.Parameter(torch.empty(768, 512))

        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        # self.classifier_image = nn.Linear(args.image_size, args.label_number)
        self.classifier_image = nn.Linear(args.text_size, args.label_number)
        # print("classifiers : ", self.classifier_fuse, self.classifier_text, self.classifier_image)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels):

        output = self.model(**inputs)
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']

        # adding normalized features
        # image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        # text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)
        # text_embeds = self.model.text_projection(text_features)
        # image_embeds = self.model.visual_projection(image_features)
        text_embeds = text_features @ self.text_projection
        image_embeds = image_features @ self.image_projection

        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        #masking visual tokens , 196 in number
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 196).to(text_features.device), inputs['attention_mask']), dim=-1) # (8,260)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 196:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        # print("fuse, text, img feature shape: ", fuse_feature.shape, "Nan check : " ,torch.isnan(fuse_feature).any())
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
