import os
import wandb
import logging
from tqdm import tqdm, trange

import numpy as np
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from data.data_set import MyDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train_clip(args, model, device, train_data, dev_data, test_data):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True, num_workers=4)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)

    clip_params = list(map(id, model.model.parameters()))
    base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    optimizer = AdamW([
            {"params": base_params},
            {"params": model.model.parameters(),"lr": args.clip_learning_rate}
            ], lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                            num_training_steps=total_steps)


    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, padding_mask, id_list = batch
            batch_image, batch_text, batch_padding_mask = image_list.to(device), text_list.to(device), padding_mask.to(device)
            labels = torch.tensor(label_list).to(device)
            batch_id = torch.tensor(id_list).to(device)

            loss, score = model(batch_image, batch_text, batch_padding_mask, batch_id, labels=labels)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        wandb.log({'train_loss': sum_loss/sum_step})
        dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, dev_data, mode='dev')
        wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

        if dev_acc > max_acc:
            max_acc = dev_acc

            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, f'model_{i_epoch}.pt'))

            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_data,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_data, mode='test')
            wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
                     'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
                     'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data, macro=False,pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size, shuffle=False, num_workers=4)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, padding_mask, id_list = t_batch
                batch_image, batch_text, batch_padding_mask = image_list.to(device), text_list.to(device), padding_mask.to(device)
                labels = torch.tensor(label_list).to(device)
                batch_id = torch.tensor(id_list).to(device)
                t_targets = labels
                loss, t_outputs = model(batch_image, batch_text, batch_padding_mask, batch_id, labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        if mode == 'test':
            wandb.log({'test_loss': sum_loss/sum_step})
        else:
            wandb.log({'dev_loss': sum_loss/sum_step})
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall
