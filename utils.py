# coding=utf-8

from random import randint, shuffle, choice 
from random import random as rand 
import math 
import numpy as np 
import torch 
import torch.utils.data 

def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i] 


# collate_fn for data_loader
def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(None)
    return batch_tensors 


def _get_word_split_index(tokens, st, end):
    split_idx = []
    i = st
    while i < end:
        if (not tokens[i].startswith('##')) or (i == st):
            split_idx.append(i)
        i += 1
    split_idx.append(end)
    return split_idx


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end 


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b


def truncate_tokens_signle(tokens_a, max_len):
    if len(tokens_a) > max_len-2:
        tokens_a = tokens_a[:max_len-2]
    return tokens_a

