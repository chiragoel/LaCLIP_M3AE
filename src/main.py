import os
import wandb
import pickle
import argparse

import torch
import random
import numpy as np
from PIL import ImageFile
from ml_collections import ConfigDict
from transformers import CLIPProcessor

from data.data_set import MyDataset
from data.data_set_original import MyDatasetOriginal

from models.model import MV_CLIPOriginal
from models.model_laclip import MV_CLIP
from models.model_mmae_laclip import MMAECLIP

from train import train_clip
from train_original import train_original

ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='MV_CLIP', type=str, help='the model name', choices=['MV_CLIP_original', 'MV_CLIP', 'MV_LaCLIP', 'MV_CLIP_MMAE', 'MV_LaCLIP_MMAE'])
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--augs', default=False, type=bool, help='Add augmented data or not')
    parser.add_argument('--replicate_mmae', default=False, type=bool, help='Transformer to replicate MMAE')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    map_location = device + ':0' if device == 'cuda' else device
    print('Augs are applied?', args.augs)

    seed_everything(args.seed)

    wandb.init(
        project="LaClip_MMSD_chirag",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.watch_called = False  
    print('Model name:', args.model)

    if args.model == 'MV_CLIP_original':
        print('In original DL')
        train_data = MyDatasetOriginal(mode='train', text_name=args.text_name, limit=None, is_augs=args.augs)
        dev_data = MyDatasetOriginal(mode='valid', text_name=args.text_name, limit=None)
        test_data = MyDatasetOriginal(mode='test', text_name=args.text_name, limit=None)
    else:
        print('In CLIP DL')
        train_data = MyDataset(mode='train', text_name=args.text_name, limit=None, is_augs=args.augs)
        dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
        test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)

    if args.model == 'MV_CLIP_original':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = MV_CLIPOriginal(args, replicate_mmae=args.replicate_mmae)
    elif args.model == 'MV_CLIP':
        model = MV_CLIP(args,map_location, device, clip_model_name='clip', replicate_mmae=args.replicate_mmae)
    elif args.model == 'MV_LaCLIP':
        model = MV_CLIP(args,map_location, device, clip_model_name='laclip', replicate_mmae=args.replicate_mmae)
    elif args.model == 'MV_CLIP_MMAE':
        model_config = ConfigDict(dict(model_type='base', drop=args.dropout_rate)) 
        model = MMAECLIP(args, device, config_updates=model_config,  layers=args.layers, num_classes=2, model_type='base', global_pool='org', clip_model_name='clip')
    elif args.model == 'MV_LaCLIP_MMAE':
        model_config = ConfigDict(dict(model_type='base', drop=args.dropout_rate)) 
        model = MMAECLIP(args, device, config_updates=model_config,  layers=args.layers, num_classes=2, model_type='base', global_pool='org', clip_model_name='laclip')
    else:
        raise RuntimeError('Error model name!')

    model.to(device)
    wandb.watch(model, log="all")
    print('Device', device)

    if args.model == 'MV_CLIP_original':
        print('in original')
        train_original(args, model, device, train_data, dev_data, test_data, processor)
    else:
        print('in CLIP')
        train_clip(args, model, device, train_data, dev_data, test_data)



if __name__ == '__main__':
    main()
