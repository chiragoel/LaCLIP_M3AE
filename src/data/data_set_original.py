import os
import json
import logging

from torch.utils.data import Dataset

import transformers
from PIL import Image
import torchvision.transforms as transforms
logger = logging.getLogger(__name__)
WORKING_PATH="/home/chirag/projects/def-sponsor00/sarcasticcodecrew/MMSD2.0/data"

class MyDatasetOriginal(Dataset):
    def __init__(self, mode, text_name, limit=None, is_augs=False):
        
        self.text_name = text_name
        self.is_augs = is_augs
        self.data = self.load_data(mode, limit)
        self.image_ids=list(self.data.keys())
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"dataset_image",str(id)+".jpg")
        if self.mode=='train' and self.is_augs==True:
            self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=21)
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
        if self.mode=='train' and self.is_augs==True:
            image_feature = self.transform(self.image_loader(id))
        else:
            image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        return text,image_feature, label, id

    def __len__(self):
        return len(self.image_ids)
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
        return text_list, image_list, label_list, id_list

