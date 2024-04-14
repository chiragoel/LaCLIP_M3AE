from transformers import CLIPModel,BertConfig,AutoConfig
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.clip.modeling_clip import CLIPModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import laclip_model_arch
from collections import OrderedDict
# from open_clip import create_model_and_transforms
from transforms import create_model_and_transforms

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
            # print("fuse hid, aal attention: ", torch.isnan(hidden_states).any(), torch.isnan(attention).any())
            # raise RuntimeError

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
        image_features = output['vision_model_output']['last_hidden_state'] #(32,50,768)
        text_feature = output['text_model_output']['pooler_output'] #(32,512)
        image_feature = output['vision_model_output']['pooler_output'] #(32,768)
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
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-B-32', # check on the patch=>224*224
            '',
            precision='amp',
            device=torch.device('mps'),
            jit=False,
            force_quick_gelu=True,
            pretrained_image=False
        )

        clip_path = "/Users/parnika./Downloads/MMSD2.0-laclip-integ-dev/laion400m_clip.pt"
        ckpt = torch.load(clip_path, map_location='cpu')
        # state_dict = OrderedDict()
        # for k, v in ckpt['state_dict'].items():
        #     state_dict[k.replace('module.', '')] = v
        self.model = model
        # self.model.eval()
        # self.model = getattr(laclip_model_arch, "CLIP_VITB32")()
        # self.model.cuda()
        self.model.load_state_dict(ckpt['state_dict'], strict=True)


        # laclip_path = "/Users/parnika./Downloads/MMSD2.0-laclip-integ-dev/laion400m_laclip.pt"
        # self.model = getattr(laclip_model_arch, "CLIP_VITB32")()
        # self.model.load_state_dict(torch.load(laclip_path), strict=False)

        #***************Need to check why freezing pretrained weights not working well or something is wrong**********#

        # self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = False

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

    # def build_model(self, state_dict: dict):
    #     vit = "visual.proj" in state_dict
    #
    #     if vit:
    #         vision_width = state_dict["visual.conv1.weight"].shape[0]
    #         vision_layers = len(
    #             [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    #         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    #         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    #         image_resolution = vision_patch_size * grid_size
    #     else:
    #         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
    #                         [1, 2, 3, 4]]
    #         vision_layers = tuple(counts)
    #         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    #         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    #         vision_patch_size = None
    #         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    #         image_resolution = output_width * 32
    #
    #     embed_dim = state_dict["text_projection"].shape[1]
    #     context_length = state_dict["positional_embedding"].shape[0]
    #     vocab_size = state_dict["token_embedding.weight"].shape[0]
    #     transformer_width = state_dict["ln_final.weight"].shape[0]
    #     transformer_heads = transformer_width // 64
    #     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    #
    #     model = self.model(
    #         embed_dim,
    #         image_resolution, vision_layers, vision_width, vision_patch_size,
    #         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    #     )
    #
    #     for key in ["input_resolution", "context_length", "vocab_size"]:
    #         if key in state_dict:
    #             del state_dict[key]

    # def forward(self, inputs, labels):
    #     # output = self.model(inputs['pixel_values'], inputs['input_ids'])  #output_attention called but not used in MV_CLIP
    #     output = self.model(inputs['pixel_values'], inputs['input_ids'],
    #                         inputs['attention_mask'])  # output_attention called but not used in MV_CLIP
    #     text_features = output['text_model_output']['last_hidden_state']
    #     image_features = output['vision_model_output']['last_hidden_state']
    #     text_feature = output['text_model_output']['pooler_output']
    #     image_feature = output['vision_model_output']['pooler_output']
    #     text_feature = self.text_linear(text_feature)
    #     image_feature = self.image_linear(image_feature)
    #
    #     text_embeds = self.model.text_projection(text_features)
    #     image_embeds = self.model.visual_projection(image_features)
    #
    #     input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
    #     attention_mask = torch.cat(
    #         (torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
    #     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    #     extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
    #     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    #     fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask,
    #                                               output_all_encoded_layers=False)
    #     fuse_hiddens = fuse_hiddens[-1]
    #     new_text_features = fuse_hiddens[:, 50:, :]
    #     new_text_feature = new_text_features[
    #         torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(
    #             torch.int).argmax(dim=-1)
    #     ]
    #
    #     new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)
    #
    #     text_weight = self.att(new_text_feature)
    #     image_weight = self.att(new_image_feature)
    #     att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1), dim=-1)
    #     tw, iw = att.split([1, 1], dim=-1)
    #     fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature
    #
    #     logits_fuse = self.classifier_fuse(fuse_feature)
    #     logits_text = self.classifier_text(text_feature)
    #     logits_image = self.classifier_image(image_feature)
    #
    #     fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
    #     text_score = nn.functional.softmax(logits_text, dim=-1)
    #     image_score = nn.functional.softmax(logits_image, dim=-1)
    #
    #     score = fuse_score + text_score + image_score
    #
    #     outputs = (score,)
    #     if labels is not None:
    #         loss_fuse = self.loss_fct(logits_fuse, labels)
    #         loss_text = self.loss_fct(logits_text, labels)
    #         loss_image = self.loss_fct(logits_image, labels)
    #         loss = loss_fuse + loss_text + loss_image
    #
    #         outputs = (loss,) + outputs
    #     return outputs

    def forward(self, inputs, labels):
        # output = self.model(inputs['pixel_values'], inputs['input_ids'])  #output_attention called but not used in MV_CLIP
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
