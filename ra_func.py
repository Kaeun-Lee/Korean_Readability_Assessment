from tqdm import tqdm
import numpy as np
from pororo import Pororo
sa = Pororo(task="sentiment", model="brainbert.base.ko.nsmc", lang="ko")

def ra_sentimentAssesment(input_sentence:str,func):
    ra_sentimentAssesment = []

    for sent in tqdm(input_sentence.split('.')):
        o = func(sent,show_probs=True)
        ra_sentimentAssesment.append(o['positive'])
    
    ra_sentimentAssesment = np.array(ra_sentimentAssesment).std() * (len( ra_sentimentAssesment)**0.5)
    return ra_sentimentAssesment
