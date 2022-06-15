import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import argparse
import yaml
import os, shutil
import numpy as np
from sklearn.metrics import roc_auc_score
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from sklearn.metrics import accuracy_score, classification_report
from ema import ExponentialMovingAverage as EMA
from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from utils import *
from datasets import *
from models import *
from scheduler import *

setup_seed(3407)
device = torch.device("cuda")

def merge_collate(batch):
    imgs = []
    labels = []
    for imgA, imgB, mtach_label, imgA_label, imgB_label in batch:
        imgs.append(imgA)
        imgs.append(imgB)
        labels.append(imgA_label)
        labels.append(imgB_label)
    return torch.stack(imgs, 0), torch.stack(labels, 0)



def train(cfg, model, loss_func, device, train_loader, optimizer, epoch, logger, logstep, writer):
    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        embeddings = model(data)

        loss = loss_func(embeddings, labels)

        loss.backward()
        optimizer.step()

        if batch_idx % logstep == 0:
            logger.info(f'[Train] Epoch {epoch} Iteration {batch_idx} : loss_circle = {loss}')
            writer.add_scalar('Loss', loss, global_step= (epoch - 1) * len(train_loader) + batch_idx)


def val(model, device, val_loader, logger, ema=None):
    model.eval()
    pred_scores, gt_scores, pred_round = [], [], []
    total_acc = 0
    for iteration, (imgA, imgB, label) in enumerate(val_loader):
        gt_scores += list(label.cpu().numpy())
        imgA = imgA.type(torch.FloatTensor).to(device)
        imgB = imgB.type(torch.FloatTensor).to(device)
        label = label.type(torch.FloatTensor)

        with torch.no_grad():
            out1, out2 = model.get_pair_embedding(imgA, imgB)

            out = torch.cosine_similarity(out1, out2)
            out_ = torch.round(out)
            pred_round += list(out_.cpu().numpy())
            pred_scores += list(out.cpu().numpy())

            acc = accuracy_score(label.cpu().numpy(), out_.cpu().numpy())
            total_acc  += acc.item()


        val_acc = total_acc / (iteration + 1)

        if iteration == 0:
            tb = PrettyTable()
            a = ['Label'] + list(label.cpu().numpy())
            b = ['Cosine'] + list(out.cpu().numpy())
            tb.add_row(a)
            tb.add_row(b)

    pred_scores, gt_scores, pred_round = np.asarray(pred_scores), np.asarray(gt_scores),  np.asarray(pred_round)
    val_auc = roc_auc_score(gt_scores, pred_scores)
    logger.info('\n' + classification_report(gt_scores, pred_round))
    return val_auc, val_acc, tb


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()

    return args


def main(args):
    with open(args.config, 'r', encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(os.path.join('runs', cfg['logname']), exist_ok=True)

    logger = get_logger(os.path.join('runs', cfg['logname'], 'trainlog.txt'))
    logger.info(cfg)

    writer = SummaryWriter(os.path.join('runs', cfg['logname']))

    data_source = load_train_data2(cfg['train_json_path'])
    num_classes = max([int(d[-1]) for d in data_source]) + 1
    logger.info(f"use train data: {cfg['train_json_path']}")
    logger.info(f'data_source len = {len(data_source)}')
    logger.info(f'num_classes = {num_classes}')
    train_datset = BiometricsClsDataset2(data_source, cfg['train_images_dir'], imgsz=cfg['imgsz'], p = 0.5, is_aug=True)
    train_sampler = RandomIdentitySampler(data_source, cfg['k'])
    train_loader = Data.DataLoader(
        dataset=train_datset,  
        batch_size=cfg['batchsize'],         
        num_workers=16,
        drop_last=False,
        sampler=train_sampler
    )


    val_dataset = BiometricsValDataset(cfg['val_json_path'],
                                        cfg['val_images_dir'], imgsz=cfg['imgsz'])
    val_loader = Data.DataLoader(
        dataset=val_dataset,  
        batch_size=10,       
        shuffle=False,     
        num_workers=4,
        drop_last=False 
    )


    num_epochs = cfg['epochs']
    model = MultiStageEmbeddingNet(stride=cfg['stride'], encoder_name=cfg['encoder_name']).to(device)
    
    if cfg['ema']:
        ema = EMA(model.parameters(), decay=0.995)
        logger.info("Use EMA")

    # distance = distances.CosineSimilarity()
    # loss_func = losses.MultiSimilarityLoss(alpha=cfg['alpha'], beta=cfg['beta'], base=float(cfg['base']), distance=distance)
    # mining_func = miners.MultiSimilarityMiner(epsilon=float(cfg['epsilon']))

    loss_func = losses.CircleLoss(m=0.4, gamma=80)

    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': float(cfg['lr'])},
        ], weight_decay=float(cfg['weight_decay']))

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-7, last_epoch=-1)
    logger.info(f"use CosineAnnealingLR ")
    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    best_val_auc, best_val_acc = 0, 0
    for epoch in tqdm(range(1, num_epochs + 1)):

        warmup_scheduler.step()
        writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step = epoch)

        train(cfg, model, loss_func, device, train_loader, optimizer, epoch, logger, cfg['logstep'], writer)

        if cfg['ema']:
            ema.update()

        if epoch < 10:
            val_auc, val_acc, tb = val(model, device, val_loader, logger)

            writer.add_scalar('val auc', val_auc, global_step = epoch)
            writer.add_scalar('val acc', val_acc, global_step = epoch)
            logger.info(f'[Val] Val ROC AUC = [{val_auc} / {best_val_auc}], round ACC = [{val_acc} / {best_val_acc}]')
            print(tb)

            if best_val_auc < val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), os.path.join('runs', cfg['logname'], 'best_auc.pth'))
                logger.info(f'[SAVE] save best weights at {epoch} epoch @ val_auc = {best_val_auc}')

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join('runs', cfg['logname'], 'best_acc.pth'))
                logger.info(f'[SAVE] save best weights at {epoch} epoch @ val_acc = {best_val_acc}')

        #if epoch >= cfg['ckpt_epoch']:
        if epoch >= 150 and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join('runs', cfg['logname'], f'ckpt_{epoch}.pth'))
            logger.info(f'[SAVE] save checkpoint {epoch} epoch weights')
        else:
            torch.save(model.state_dict(), os.path.join('runs', cfg['logname'], 'last.pth'))
            logger.info('[SAVE] save last weights')



if __name__ == '__main__':
    args = parse_args()
    main(args)