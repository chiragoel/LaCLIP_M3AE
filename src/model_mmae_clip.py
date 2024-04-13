from m3ae_model import *
from transformers import CLIPModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MMAECLIP(nn.Module):
    def get_default_config(self, updates=None):
        config = ConfigDict()
        config.model_type = 'small'
        config.emb_dim = 1024
        config.dec_emb_dim = 512
        config.depth = 24
        config.dec_depth = 8
        config.num_heads = 16
        config.dec_num_heads = 16
        config.mlp_ratio = 4

        config.output_head_depth = 0
        config.att_drop = 0.0
        config.drop = 0.0
        config.drop_path = 0.0

        config.use_type_embedding = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        if config.model_type is not None:
            get_transformer_by_config(config.model_type, config)

        return config

    def __init__(self, args, device, config_updates=None,  layers=3, num_classes=2, model_type='base'):
        super(MMAECLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = self.get_default_config(config_updates)
        self.device = device
        self.num_classes = num_classes
        
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
        self.text_projection = nn.Linear(args.text_size, args.image_size)
        self.image_projection = nn.Linear(args.image_size, args.image_size)
        
        if self.config.use_type_embedding:
            self.encoder_image_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            ).to(device)
            self.encoder_text_type_embedding = nn.Parameter(
                torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
            ).to(device)

        self.cls_token = nn.Parameter(
            torch.empty(1, 1, self.config.emb_dim).normal_(0.02)
        ).to(device)

        self.encoder = self.load_embed_model(model_type, device, config_updates, layers)
        
        self.classifier_fuse = nn.Linear(args.image_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()

        
    def load_embed_model(self, model_type, device, config_updates, layers):
        if model_type == 'base':
            embed_model_load_path = './torch_weights/m3ae_base.pth'
        elif args.model_type == 'small':
            embed_model_load_path = './torch_weights/m3ae_small.pth'
        else:
            embed_model_load_path = './torch_weights/m3ae_large.pth'

        map_location = device + ':0' if device == 'cuda' else device
        encoder_model = MaskedMultimodalAutoencoder(30522, device, config_updates)
        encoder_model.to(device)
        check_point_embed_model = torch.load(embed_model_load_path, map_location=map_location)
        encoder_model.load_state_dict(check_point_embed_model, strict=False)
        if layers is None:
            return encoder_model
        transfer_weights = nn.ModuleList([encoder_model.encoder.blocks[i] for i in range(layers)])
        transfer_weights.to(device)
        

        embed_model = Transformer(
            emb_dim=self.config.emb_dim,
            depth=layers,
            att_drop=self.config.att_drop,
            drop=self.config.drop,
            drop_path=self.config.drop_path,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            device=self.device
        )
        embed_model.to(device)
        embed_model.blocks.load_state_dict(transfer_weights.state_dict())
        return embed_model

    def get_type_embedding(self, name):
        if self.config.use_type_embedding:
            return {
                'encoder_image_type_embedding': self.encoder_image_type_embedding,
                'encoder_text_type_embedding': self.encoder_text_type_embedding,
            }[name]
        else:
            return 0.0

    def forward(self, inputs, labels, deterministic=False, global_pool=True):
        output = self.model(**inputs,output_attentions=True)
        
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)
        
        # Add projection layers text (512->768) and image (768->768)
        text_features = self.text_projection(output['text_model_output']['last_hidden_state']) #Remove CLS token
        image_features = self.image_projection(output['vision_model_output']['last_hidden_state'][:,1:,:])
        # print(text_features.shape, image_features.shape)
        batch_size = image_features.shape[0]
        
        #TODO: Resolve CLS token Discrepancy
        cls_token = self.cls_token.expand(batch_size, 1, self.config.emb_dim)
        input_tensors = [cls_token]
        padding_masks = [torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)]
        
        input_tensors.append(image_features)
        padding_masks.append(torch.zeros((batch_size, image_features.shape[1]), dtype=torch.float32).to(self.device))

        
        input_tensors.append(text_features)
        #Do 1-padding mask because the attention class will deselect all with weights >0. So tokens that are present in text should be 0
        padding_masks.append((1.0 - inputs['attention_mask']).to(self.device))

        x = torch.cat(input_tensors, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        # print('pm', padding_mask.shape, x.shape)
        x = self.encoder(x, deterministic, padding_mask)
        # The first token (CLS token) will have the combined info or we can global pool and aggregate info or we can also do it similar to MV_CLIP model where we do weight based aggregation
        if global_pool:
            fuse_feature = x[:, 1:, :].mean(axis=1)  # global pool without cls token
        else:
            fuse_feature = x[:, 0]

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