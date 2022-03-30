import os
import requests
from tika import parser
from tqdm import tqdm 
import pandas as pd
import warnings 
import pickle 
warnings.filterwarnings(action='ignore')

error = 0
df = pd.DataFrame()
for file_num in tqdm(range(217,1885)):
    try:
        url = f'https://www.ksa.or.kr/krca/ksi/1/{file_num}/download.do'
        file = requests.get(url, allow_redirects=True,verify=False)
        open(f'report_{file_num}.pdf', 'wb').write(file.content)

        raw = parser.from_file(f'report_{file_num}.pdf')
        content = raw['content']
        row = pd.DataFrame({"file_url":url,
                            "file_content":content,
                            "file_length":len(content)}, index=[file_num])
        df = pd.concat([df,row],axis=0)
        os.remove(f'report_{file_num}.pdf')
    except:
        error+=1
        print(f'{error}th occured!')
        
with open("Korean Sustainability Report RA Dataset.bin","wb") as fw:
    pickle.dump(df, fw)

import pandas as pd
from tqdm import tqdm
from pororo.tasks.grammatical_error_correction import PororoGecFactory,PororoTransformerGec
import math 

spacing = PororoGecFactory(task="gec",lang="ko",model="charbert.base.ko.spacing").load('cuda:0')


def tp_sentenceCleanse(input_sentence:str, func, upper_bound=512):
    output_sentence_space = ''
    tot_block = math.ceil(len(input_sentence)/upper_bound)
    for _ in range(tot_block):
        s = _ * upper_bound
        e = (_+1) * upper_bound
        output_sentence = input_sentence[s:e]
        output_sentence = func(output_sentence)
        output_sentence_space += output_sentence
    return output_sentence_space
