import open_clip.src.open_clip as oc
import torch


model = oc.factory.create_model('ViT-B-32')

print('Model Info')
print(model)
for n,p in model.named_parameters():
    print(n)
    
chkt = torch.load('./laclip_model/laion400m_laclip.pt')
print('Chkt ----------------------------------------------------------------')
print(chkt['state_dict'].keys())
result = model.load_state_dict(chkt['state_dict'], strict=False)
print('Result------------------------------')
print(result)