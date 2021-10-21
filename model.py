# coding=utf-8 
# from __future__ import * 

import copy 
import math 
import logging 
import numpy as np 

import torch
import torch.nn.functional as F 
from torch.nn.modules.loss import _Loss 
from transformers.modeling_utils import PreTrainedModel 

from configuration import PVTConfig 
from transformers.modeling_bert import load_tf_weights_in_bert, BertPooler, BertIntermediate, \
    BertOutput, BertPredictionHeadTransform, BertSelfOutput, BertLMPredictionHead, BertOnlyMLMHead, \
    BertOnlyMLMHead, BertEmbeddings, BertOnlyNSPHead


logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm 




