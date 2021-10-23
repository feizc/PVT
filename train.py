# coding=utf-8 

import os 
import logging 
import json 
import argparse 

import numpy as np 
import torch 
from torch.utils.data import RandomSampler 
from torch.utils.data.distributed import DistributedSampler 
import torch.distributed as dist 

from tokenizer import PVTTokenizer, WhitespaceTokenizer 
from model import PVTConfig, PVTForSeq2Seq 
from transformers import AdamW, get_linear_schedule_with_warmup 

import utils 



def main(): 
    print('aa')

if __name__ == '__main__':
    main()