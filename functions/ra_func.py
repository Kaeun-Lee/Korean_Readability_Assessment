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
    num_sentences = len(input_sentence.split('.'))  # 문장 수
    input_sentence_tokenzied = func(input_sentence)    # 토큰화

    num_letters = 0  # 글자 수
    num_long_words = 0  # n음절 이상인 단어 수
    num_words = 0  # 단어 수

    for word in input_sentence_tokenzied:   # 단어 수는 빈칸(" ")으로 구분
        word = re.sub(r'▁', '', word)       # _ 제거
        word = re.sub(r'\.', '', word)      # . 제거
        
        if len(word) == n:
            num_long_words += 1
        if len(word) >= 2:
            num_letters += len(word)
            num_words += 1

    ASL = num_words / num_sentences  # 평균 문장 길이
    fog_index = (ASL + (num_long_words / num_words) * 100) * 0.4
    
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
   
   
def ra_biRSRS_batch(input_batch,model,tokenizer,device):
    model.eval()
    mask = tokenizer.mask_token_id
    pad  = tokenizer.pad_token_id

    input_ids = tokenizer.batch_encode_plus(input_batch.split('.'),
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt')['input_ids'].to(device)
    batch_val = []                                       
    for sent in input_ids:
        sent_len = (sent!=0).sum()
        sent_ids = sent[:sent_len]
        val=[]
        for i in range(1,sent_ids.shape[0]-1):
            target = sent_ids[i]
            prompt = torch.clone(sent_ids)
            prompt[i] = mask
            o = model(prompt.reshape(-1,1))['logits']
            o = torch.nn.functional.softmax(o,dim=2)
            val.append(float(o[i,0,target].cpu().detach().numpy()))

        val = sum([i*_ for i,_ in enumerate(sorted(val))]) / sent_len
        batch_val.append(val.cpu().numpy())
    return sum(batch_val)/input_ids.shape[0]
