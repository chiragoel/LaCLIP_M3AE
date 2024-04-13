from torch.utils.data import Dataset
import logging
import os
import torch
from PIL import Image
import json
import transformers
import torchvision.transforms as transforms
from laclip_tokenizer import SimpleTokenizer
logger = logging.getLogger(__name__)
WORKING_PATH="/home/chirag/projects/def-sponsor00/sarcasticcodecrew/MMSD2.0/data"

class MyDataset(Dataset):
    def __init__(self, mode, text_name, limit=None, is_augs=True):
        
        self.text_name = text_name
        self.mode = mode
        self.data = self.load_data(mode, limit)
        self.image_ids=list(self.data.keys())
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"dataset_image",str(id)+".jpg")
        self.tokenizer = SimpleTokenizer()
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.ToTensor(),
                normalize
            ])
        self.val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

    
    def load_data(self, mode, limit):
        cnt = 0
        data_set=dict()
        if mode in ["train"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            for data in datas:
                if limit != None and cnt >= limit:
                    break

                image = data['image_id']
                sentence = data['text']
                label = data['label']
 
                if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                    data_set[int(image)]={"text":sentence, 'label': label}
                    cnt += 1
                    
        
        if mode in ["test","valid"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            for data in datas:
                image = data['image_id']
                sentence = data['text']
                label = data['label']

                if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                    data_set[int(image)]={"text":sentence, 'label': label}
                    cnt += 1
        return data_set


    def image_loader(self,id):
        return Image.open(self.data[id]["image_path"])
    def text_loader(self,id):
        return self.data[id]["text"]


    def __getitem__(self, index):
        id=self.image_ids[index]
        
        text = self.text_loader(id)
        toks, text = self.tokenizer(str(text))
        padding_mask = torch.zeros(77)
        padding_mask[len(toks):] = 1
        
        if self.mode == 'train':
            image_feature = self.train_transform(self.image_loader(id))
        else:
            image_feature = self.val_transform(self.image_loader(id))
        
        
        label = self.data[id]["label"]
        
        return text,image_feature, label, padding_mask, id

    def __len__(self):
        return len(self.image_ids)
    # @staticmethod
#     def collate_func(batch_data):
#         batch_size = len(batch_data)
 
#         if batch_size == 0:
#             return {}

#         text_list = []
#         image_list = []
#         label_list = []
#         id_list = []
#         for instance in batch_data:
#             text_list.append(instance[0])
#             image_list.append(instance[1])
#             label_list.append(instance[2])
#             id_list.append(instance[3])
#         return text_list, image_list, label_list, id_list

