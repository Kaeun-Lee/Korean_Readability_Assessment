from tqdm import tqdm
import numpy as np
import re
from pororo.tasks.sentiment_analysis import PororoSentimentFactory
from pororo.tasks.tokenization import PororoTokenizationFactory

tk_func = PororoTokenizationFactory(task="tokenization", lang="ko", model="bpe32k.ko")

 
def ra_sentimentAssessment(input_sentence:str, func):
    ra_sentimentAssesment = []

    for sent in input_sentence.split('.'):  # 문장 단위로 쪼개기
        if len(sent) > 512:                 # pororo 라이브러리 특성상 최대 토큰 수 512, 이상으로 길 경우 512에서 truncate
            sent = sent[:512]
        o = func(sent,show_probs=True)         
        ra_sentimentAssesment.append(o['positive']) # o['negative'] = 1 - o['positive'] 라 내재 변동성 계산하면 positive/negative 둘 중 무엇을 사용해도 값이 동일함
    
    ra_sentimentAssesment = np.array(ra_sentimentAssesment).std() * (len( ra_sentimentAssesment)**0.5)
    return ra_sentimentAssesment

def ra_sentimentAssessment_batch(input_batch:list, device='cpu'): # ra_sentimentAssessment를 batch 단위로 활용할 때 더욱 빠르게 사용할 수 있게 해줌
    sa_func = PororoSentimentFactory(task="sentiment", lang="ko", model="brainbert.base.ko.nsmc").load(device)
    pbar = tqdm(input_batch)
    output = []
    for _ in pbar:
        score = ra_sentimentAssessment(_, sa_func)
        output.append(score)
        pbar.set_description(f"Current Score: {sum(output)/len(output)}")
    return output

def ra_fogIndex(input_sentence:str, n:int, device='cpu'):
    func = tk_func.load(device)
    input_sentence_length = len(input_sentence.split('.'))  # 문장 수
    input_sentence_tokenzied = func(input_sentence)    # 토큰화

    num_letter = 0  # 글자 수
    long_words = 0  # n음절 이상인 단어 수
    total_word = 0  # 단어 수

    for word in input_sentence_tokenzied:   # 단어 수는 빈칸(" ")으로 구분
        word = re.sub(r'▁', '', word)       # _ 제거
        word = re.sub(r'\.', '', word)      # . 제거
        
        if word != '':
            if len(word) == n:
                long_words += 1
            if len(word) >= 2:
                num_letter += len(word)
                total_word += 1

    ASL = total_word / input_sentence_length  # 평균 문장 길이
    fog_index = (ASL + (long_words / total_word) * 100) * 0.4
    
    return fog_index

def ra_fogIndex_batch(input_batch:list, n:int, device='cpu'): # ra_fogIndex를 batch 단위로 활용할 때 더욱 빠르게 사용할 수 있게 해줌
    tk_func = PororoTokenizationFactory(task="tokenization", lang="ko", model="bpe32k.ko").load(device)
    pbar = tqdm(input_batch)
    output = []
    for _ in pbar:
        score = ra_fogIndex(_, n, tk_func)
        output.append(score)
        pbar.set_description(f"Current Score: {sum(output)/len(output)}")
    return output