import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from m3ae_model import *
from train_m3ae import train
from data_set_m3ae import MyDataset
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import wandb
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--global_pool', default=False, type=bool, help='Use CLS token or pool all the outputs')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=77, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    # parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    # parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--model_type', default='small', type=str, help='Model type - small, base, large')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--use_type_embedding', default=True, type=bool, help='Use special image and text tokens for addition')
    parser.add_argument('--output_dir', default='../output_dir_m3ae/', type=str, help='the output path')
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    print(device)

    seed_everything(args.seed)

    wandb.init(
        project="MMAE_Team",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.watch_called = False
    
    print('Loading Data')

    train_data = MyDataset(mode='train', text_name=args.text_name, text_dim=args.text_size, image_dim=args.image_size, limit=None, is_augs=True)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, text_dim=args.text_size, image_dim=224, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, text_dim=args.text_size, image_dim=224, limit=None)

    print('Loading Pretrained weights')
    
    
    model_config = ConfigDict(dict(model_type=args.model_type, drop=args.dropout_rate)) 
    
    model = MMAEBase(text_vocab_size=30522, device=device, config_updates=model_config, img_embed_dim=3072,  layers=args.layers, num_classes=2, model_type=args.model_type)

    model.to(device)
    wandb.watch(model, log="all")

    train(args,model, device, train_data, dev_data, test_data)



if __name__ == '__main__':
    print('Started')
    main()
