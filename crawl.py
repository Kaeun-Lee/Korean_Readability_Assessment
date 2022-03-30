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
from pororo import Pororo
spacing = Pororo(task="gec", lang="ko")

df = pd.read_excel('Korean Readability Assessment.xlsx')

def ko_spacing(sent):
    try:
        if len(sent) <512:
            return spacing(sent)
        else: 
            return spacing(sent[:512]) + spacing(sent[:512:])
    except:
        return '-'
    
file_contentCleansed = []

for content in tqdm(df['file_content'].values):
    content = content.split('. \n')[1:]
    content = ' ' .join([_.replace('\n','') for _ in content])
    content = [ko_spacing(_+'.') for _ in content.split('.') if _.endswith('다')]
    content = ''.join(content)
    file_contentCleansed.append(content)
    
df['file_contentCleansed'] = file_contentCleansed 
df.to_excel('B_Korean Readability Assessment.xlsx')
