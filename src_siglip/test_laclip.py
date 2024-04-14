from transformers import CLIPModel,BertConfig, CLIPConfig
import torch
import laclip_model_arch
import model


laclip_path = "/home/zeus/MMSD2.0/pretrained_models/laion400m_laclip.pt"

# # model = CLIPModel.from_pretrained(laclip_path)

laclip_model = getattr(laclip_model_arch, "CLIP_VITB32")()
# print("Model : ", laclip_model)
checkpoint = torch.load(laclip_path, map_location='cpu')
# model_weight = torch.load(laclip_path)
print(checkpoint['state_dict']['text_projection'].shape)
result = laclip_model.load_state_dict(checkpoint['state_dict'], strict=False)
# print("response after loading wts", result)
print(laclip_model, type(laclip_model))
print("------------------")
# config = CLIPConfig(
#     num_classes=0,  # Set appropriate value for your model
#     hidden_dim=512,  # Set appropriate value for your model
#     vision_width=224,  # Set appropriate value for your model
#     vision_layers=12,  # Set appropriate value for your model
#     text_width=77,  # Set appropriate value for your model
#     text_layers=12,  # Set appropriate value for your model
#     zero_init_last_layer=True,  # Set appropriate value for your model
#     use_conv=False,  # Set appropriate value for your model
# )
from transformers import AutoConfig
# Load the configuration of the pre-trained CLIP model
config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print(type(clip_model.model))
# print(clip_model, type(clip_model))

# with open("clip_config.txt", "w") as f:
#     f.write(str(config))
# with open("model_config_compare.txt", "w") as f:
#     f.write(str(clip_model))
#     f.write("=======================================================================================")
#     f.write(str(laclip_model))

# Print the configuration
# print(config)

# model = CLIPModel.from_pretrained(pretrained_model_name_or_path=laclip_path, config=config)
# print("yes done")

# model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # print('normal clip config details: ', model_clip.config)
# print(type(model_clip.config), type(config))
# assert model_clip.config == config #,print("config not same")