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
        self.is_augs = is_augs

        self.tokenizer = SimpleTokenizer()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if self.mode=='train' and self.is_augs==True:
            self.transform = transforms.Compose([
            transforms.CenterCrop(size=(self.image_dim, self.image_dim)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * self.image_dim))
            transforms.ToTensor(),
            
            ])
        elif self.mode=='train':
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=(self.image_dim, self.image_dim)),
                transforms.ToTensor(),
                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=(self.image_dim, self.image_dim)),
                transforms.ToTensor(),
                
            ])

    
    def load_data(self, mode, limit):
        cnt = 0
        data_set=dict()
        if mode in ["train"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            f2= open(os.path.join('/home/chirag/projects/def-sponsor00/chirag/CLIP-MA-MMSD/data/augmented_data (1).json'),'r',encoding='utf-8')
            datas = json.load(f1)
            for data in datas:
                if limit != None and cnt >= limit:
                    break

                image = data['image_id']
                sentence = data['text']
                label = data['label']
 
                if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                    data_set[int(cnt)]={"text":sentence, 'label': label, 'image_path': os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")}
                    cnt += 1
            if self.is_augs:
                datas_augs = json.load(f2)
                for data in datas_augs:
                    if limit != None and cnt >= limit:
                        break

                    image = data['image_id']
                    sentence = data['text']
                    label = data['label']
    
                    if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                        data_set[int(cnt)]={"text":sentence, 'label': label, 'image_path': os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")}
                        cnt += 1
                    
        
        if mode in ["test","valid"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            for data in datas:
                image = data['image_id']
                sentence = data['text']
                label = data['label']

                if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                    data_set[int(cnt)]={"text":sentence, 'label': label, 'image_path': os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")}
                    cnt += 1
        return data_set


    def image_loader(self,id):
        return Image.open(self.data[id]["image_path"])
    def text_loader(self,id):
        return self.data[id]["text"]


    def __getitem__(self, index):
        id=index
        
        text = self.text_loader(id)
        toks, text = self.tokenizer(str(text))
        padding_mask = torch.zeros(77)
        padding_mask[len(toks):] = 1
        
        image_feature = self.transform(self.image_loader(id))
        image_feature = self.normalize(image_feature*2 - 1)
       
        
        label = self.data[id]["label"]
        
        return text,image_feature, label, padding_mask, id

    def __len__(self):
        return len(self.data.keys())