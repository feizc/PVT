# coding=utf-8 

import os 
import logging 
import json 
import argparse 
import random 

import numpy as np 
import torch 
from torch.utils.data import RandomSampler 
from torch.utils.data.distributed import DistributedSampler 
import torch.distributed as dist 
from transformers import AdamW, get_linear_schedule_with_warmup 
from tqdm import tqdm, trange 


from tokenizer import PVTTokenizer, WhitespaceTokenizer 
from model import PVTConfig, PVTForSeq2Seq 
from dataset import Preprocess4Seq2seq, Seq2SeqDataset 
from utils import batch_list_to_batch_tensors


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


seed = 2021 
MODEL_CLASSES = {'PVT': (PVTConfig, PVTForSeq2Seq, PVTTokenizer)}
MODEL_PATH = './model' 
DATA_PATH = './data' 
num_train_epochs = 6 
gradient_accumulation_steps = 5 



def main(): 

    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--data_dir", default='data/train.txt', type=str, 
                        help="The input data dir. ") 
    parser.add_argument("--log_dir", default='log', type=str,
                        help="The output directory where the log will be written.")
    
    args = parser.parse_args() 

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    logger.info("device: {}".format(device)) 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 

    config_class, model_class, tokenizer_class = MODEL_CLASSES['PVT'] 
    config = config_class.from_pretrained(MODEL_PATH) 
    tokenizer = tokenizer_class.from_pretrained(MODEL_PATH, do_lower_case=True)
    
    logger.info('Loading Train Dataset')
    bi_uni_pipeline = [Preprocess4Seq2seq(max_pred=20, mask_prob=0.2, 
                        vocab_words=list(tokenizer.vocab.keys()), 
                        indexer=tokenizer.convert_tokens_to_ids, 
                        tokenizer=tokenizer)]
    
    file_path = os.path.join(DATA_PATH, 'train_data.json') 
    train_dataset = Seq2SeqDataset(file=file_path, batch_size=4, tokenizer=tokenizer, 
                                    max_len=128, bi_uni_pipeline=bi_uni_pipeline)
    
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=train_sampler,
                                                       num_workers=8, collate_fn=batch_list_to_batch_tensors, 
                                                       pin_memory=False)
    t_total = int(len(train_dataloader) * num_train_epochs / gradient_accumulation_steps)

    logger.info('Prepare Model') 
    model_dict_path = os.path.join(MODEL_PATH, 'pytorch_model.bin')
    model_dict = torch.load(model_dict_path, map_location='cpu')
    model = model_class.from_pretrained(MODEL_PATH, state_dict=model_dict, config=config) 

    model.to(device) 

    logger.info('Prepare Optimizer')  
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*t_total, num_training_steps=t_total) 
    
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache() 

    logger.info("***** Running training *****") 
    model.train() 
    start_epoch = 1 
    for i_epoch in trange(start_epoch, num_train_epochs+1, desc='Epoch'): 
        iter_bar = tqdm(train_dataloader, desc="Iter (loss=X.XXX)") 
        for step, batch in enumerate(iter_bar): 
            batch = [t.to(device) if t is not None else None for t in batch]
            input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
            masked_lm_loss = model(input_ids, segment_ids, input_mask, lm_label_ids,
                                   masked_pos=masked_pos, masked_weights=masked_weights)
            loss = masked_lm_loss
            iter_bar.set_description('Iter (loss=%5.3f)' % loss.item()) 
            if gradient_accumulation_steps > 1: 
                loss = loss / gradient_accumulation_steps 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  

            if (step + 1) % gradient_accumulation_steps == 0: 
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()

            break 
        break 


if __name__ == '__main__':
    main()