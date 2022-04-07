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