!pip install transformers==4.11.0

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse
import warnings

import copy

#KoBigBird
from transformers import AutoModel, AutoTokenizer
from transformers.models.big_bird.modeling_big_bird import BigBirdLayer, BigBirdModel
model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")  # BigBirdModel
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
logSoftmax=torch.nn.LogSoftmax(dim=1).to(device)
NLLLos=torch.nn.NLLLoss().to(device)
max_length=4096

def encode(tokenizer, text_sentence, add_special_tokens=True):
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def ra_pplCalculation(input_sentence:str):
    input_sentence_length=len(input_sentence.split('.')) #문장 수
    sentences=input_sentence.split('.') #for each text is split into the sentences
    rsrs_=0 #각 문장 별 rsrs 값

    for sentence in sentences:
      words=sentence.split(' ')
      sen_length = len(words)
      wnll=[] #word negative log likelihood
      for i in range(2,sen_length):
        try:
            sub_sentence=" ".join(words[:i])+' '+tokenizer.mask_token #mask the token
            input_ids_to,mask_idx = encode(tokenizer,sub_sentence,True) # tokenize the sentences & mask token index
            begin_loc=max(i-max_length,0)
            end_loc=min(i,sen_length) ##문장 내 next word prediction

            input_ids=input_ids_to[begin_loc:end_loc-1].to(device)
            target_ids=input_ids.clone()
            target_ids[target_ids!=4]=-100 #only calculate loss on masked token
            with torch.no_grad():
              lm_pred=model(input_ids,target_ids)
              output=NLLLos(logSoftmax(torch.unsqueeze(lm_pred[0][0,mask_idx,:],0)),torch.tensor([torch.argmax(lm_pred[1][0][mask_idx]).item()]).to(device))
            wnll.append(output)
        except:
            sen_length-=1
      rsrs=sum([np.sqrt(i+1)*j for i,j in enumerate(sorted(wnll))])/sen_length
      rsrs_+=rsrs
    return rsrs_/input_sentence_length