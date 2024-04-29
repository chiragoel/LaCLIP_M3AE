from torch.utils.data import Dataset
import transformers
import torchvision.transforms as transforms
import logging
import os
import torch
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)
WORKING_PATH="/home/chirag/projects/def-sponsor00/sarcasticcodecrew/MMSD2.0/data"

class MyDataset(Dataset):
    def __init__(self, mode, text_name, text_dim=256, image_dim=224, limit=None, is_augs=False):
        self.text_name = text_name
        self.data = self.load_data(mode, limit)
        self.image_ids=list(self.data.keys())
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"dataset_image",str(id)+".jpg")
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_dim = text_dim
        self.image_dim = 224
        self.mode = mode 
        if self.mode=='train' and is_augs==True:
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(self.image_dim, self.image_dim), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        elif self.mode=='train':
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(self.image_dim, self.image_dim), scale=(0.8, 1.0)),
            transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(self.image_dim, self.image_dim), scale=(0.8, 1.0)),
            transforms.ToTensor()
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
        
        image_feature = self.image_loader(id)
        image_feature = self.transform(image_feature)
        # image_feature = (image_feature-image_feature.mean([1,2]).unsqueeze(1).unsqueeze(1))/image_feature.std([1,2]).unsqueeze(1).unsqueeze(1)
       
        text = self.text_loader(id)
        encoded_text = self.tokenizer(text,
                                 padding="max_length",
                                 truncation=True,
                                 max_length=77,
                                 return_tensors="np",
                                 add_special_tokens=False,
                                )
        tokenized_caption = torch.from_numpy(encoded_text["input_ids"][0])[None, ...]
        padding_mask = 1.0 - encoded_text["attention_mask"][0].astype(np.float32)
        
        label = self.data[id]["label"]
        # print(tokenized_caption.shape, padding_mask.shape, image_feature.shape)
        
        return tokenized_caption, padding_mask, image_feature, label, id

    def __len__(self):
        return len(self.image_ids)
#     @staticmethod
#     def collate_func(batch_data):
#         batch_size = len(batch_data)
 
#         if batch_size == 0:
#             return {}

#         text_list = []
#         image_list = []
#         label_list = []
#         id_list = []
#         padding_mask_list = []
#         for instance in batch_data:
#             text_list.append(instance[0])
#             image_list.append(instance[1])
#             label_list.append(instance[2])
#             id_list.append(instance[3])
#         return text_list, image_list, label_list, id_list
