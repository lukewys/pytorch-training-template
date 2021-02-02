import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from data_handling import get_data_loader
# from model import XXX

import itertools
import numpy as np
import os
import sys
import logging
# import wandb

from utils import print_hparams, set_seed
from model import DummyModel

from hparams import hparams as hp
from tensorboardX import SummaryWriter

import argparse

parser = argparse.ArgumentParser(description='hparams for model')

device = torch.device("cuda" if hp.cuda else "cpu")

set_seed(hp.seed)


def train(training_data):
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch = 0
    for src, tgt in training_data:

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(src)  # change according to model

        loss = criterion(output, tgt)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad)
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar('Train/loss', loss.item(), (epoch - 1) * len(training_data) + batch)

        batch += 1

        if batch % hp.log_interval == 0 and batch > 0:
            mean_loss = total_loss / hp.log_interval
            elapsed = time.time() - start_time
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            logging.info(
                f'| epoch {epoch:3d} | {batch:5d}/{len(training_data):5d} batches | lr {current_lr:02.2e} | '
                f'ms/batch {(elapsed * 1000 / hp.log_interval):5.2f} | loss {mean_loss:5.4f}')
            total_loss = 0
            start_time = time.time()


def eval(evaluation_data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in evaluation_data:
            src = src.to(device)
            # XXX

        loss_mean = XXX
        writer.add_scalar(f'Eval/loss', loss_mean, epoch)
        msg = f'eval: {loss_mean:2.4f}'
        logging.info(msg)


if __name__ == '__main__':
    # parser.add_argument('--decoder_nlayers', type=int, default=hp.decoder_nlayers)
    # parser.add_argument('--nhead', type=int, default=hp.nhead)
    # parser.add_argument('--nhid', type=int, default=hp.nhid)
    # parser.add_argument('--training_epochs', type=int, default=1000)
    # parser.add_argument('--load_pretain_emb', action='store_true')
    # parser.add_argument('--emb_bind', action='store_true')
    # parser.add_argument('--spec_augmentation', action='store_true')
    # parser.add_argument('--label_smoothing', action='store_true')
    # parser.add_argument('--tagging', action='store_true')
    # parser.add_argument('--encoder_type', type=str, default='resnet2d')
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(hp, k, v)

    # TODO：add https://github.com/PhilJd/contiguous_pytorch_params

    model = DummyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    criterion = XX

    now_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
    log_dir = 'logs/logs_{model}/{time}'.format(time=now_time, model=args.name if args.name else model.model_type)

    if hp.restore_path:
        model = torch.load(hp.restore_path)
        log_dir = os.path.dirname(hp.restore_path)

    writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, 'train.log')

    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        format='%(asctime)s - %(levelname)s: %(message)s',  # 日志格式
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)]
                        )

    training_data = get_data_loader()
    evaluation_data = get_data_loader()

    logging.info(str(model))

    logging.info(str(print_hparams(hp)))

    logging.info('Data loaded!')
    logging.info('Data size: ' + str(len(training_data)))

    logging.info('Total Model parameters: ' + f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    epoch = int(os.path.basename(hp.restore_path).replace('.pt', '')) if hp.restore_path else 1

    if hp.mode == 'train':
        while epoch < hp.training_epochs + 1:
            epoch_start_time = time.time()
            train(training_data)
            scheduler.step()
            eval(evaluation_data)
            if epoch % hp.checkpoint_save_interval == 0:
                torch.save(model, '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            epoch += 1

    elif hp.mode == 'eval':
        eval(evaluation_data)
