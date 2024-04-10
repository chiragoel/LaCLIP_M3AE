import os
from data_set_m3ae import MyDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import wandb
import numpy as np
import torch.nn.functional as F
import einops

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            # lr=1e-3,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8,
            # beta1=None,
            lr=None,
            weight_decay=args.weight_decay,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )
        scheduler = AdafactorSchedule(optimizer)
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')


    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        
        model.to(device)
        model.train()

        for step, batch in enumerate(iter_bar):
            batch_tokenized_caption, batch_padding_mask, batch_image_feature, batch_label, batch_id = batch
            batch_image_feature = einops.rearrange(batch_image_feature,
                                         'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                         p1=32,
                                         p2=32)
            batch_tokenized_caption, batch_padding_mask, batch_image_feature = batch_tokenized_caption.to(device), batch_padding_mask.to(device), batch_image_feature.to(device)
            labels = torch.tensor(batch_label).to(device)

            probs, feats = model(batch_image_feature, batch_tokenized_caption, batch_padding_mask, deterministic=False, features=True)
            # print(feats, labels)
            loss = F.cross_entropy(probs, labels)
            # print(loss)
            # break
            # Add loss function
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

            path_to_save = os.path.join(args.output_dir, 'MMAE_base')
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, f'model{i_epoch}_{step}.pth'))
            
            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_data,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_data, mode='test')
            wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
                     'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
                     'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data, macro=False,pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size,shuffle=False)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                batch_tokenized_caption, batch_padding_mask, batch_image_feature, batch_label, batch_id = t_batch
                batch_image_feature = einops.rearrange(batch_image_feature,
                                         'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                         p1=32,
                                         p2=32)
                batch_tokenized_caption, batch_padding_mask, batch_image_feature = batch_tokenized_caption.to(device), batch_padding_mask.to(device), batch_image_feature.to(device)
                labels = torch.tensor(batch_label).to(device)
                
                t_targets = labels
                probs, t_outputs = model(batch_image_feature, batch_tokenized_caption, batch_padding_mask, deterministic=False, features=True)
                # print('Testing', probs, t_outputs, labels)
                loss = F.cross_entropy(probs, labels)
                #define loss and t_target and t_output
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
