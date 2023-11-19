#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dehamer_model import dehamer
from argparse import ArgumentParser

from train_data_aug import TrainData ############# 
from val_data_train import ValData_train
from utils import validation_PSNR, generate_filelist
# from val_data import ValData

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of dehamer from Guo et al. (2022)')

    # Data parameters
    parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'dense', 'indoor','outdoor'], default='NH')
    parser.add_argument('-t', '--train_NH-dir', help='training set path', default='./../data/NH-Haze/train_NH')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/NH-Haze/valid_NH')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts/NH')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--ckpt-load-path', help='start training with a pretrained model',default='./../ckpts/NH/PSNR2066_SSIM06844.pt')
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train_NH-size',nargs='+', help='size of train_NH dataset',default=[128,128], type=int)
    parser.add_argument('-vs', '--valid-size',nargs='+', help='size of valid dataset',default=[128,128], type=int)

    # Training hyperparameters 
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=8, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', default=True, action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true') 
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)#224
 
    return parser.parse_args()

 
if __name__ == '__main__':    
    
    
    # Parse training parameters  
    params = parse_args()   
    # import pdb;
    # pdb.set_trace()
# --- Load training data and validation/test data --- #
    train_loader = DataLoader(TrainData(params.dataset_name,params.train_NH_size, params.train_NH_dir), batch_size=params.batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(ValData_train(params.dataset_name,params.valid_size,params.valid_dir), batch_size=params.batch_size, shuffle=False, num_workers=12)

    # Initialize model and train_NH
    dehamer = dehamer(params, trainable=True)
    dehamer.train(train_loader, valid_loader)






