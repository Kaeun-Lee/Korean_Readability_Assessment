# Korean_Readability_Assessment

This repo contains the resources for implementing the paper "Zero-Shot Readability Assessment of Korean ESG Reports using BERT."(pending for results).

## Data 

The paper constructs a small binary dataset for benchmarking purposes.  
The dataset is consisted of text, text_length and label(ESG, NEWS). ESG label stands for Korean ESG reports collected from [KSA](https://www.ksa.or.kr/ksa_kr/index.do), and NEWS label stands for Korean Finance News inherited from the [KLUE Dataset](https://klue-benchmark.com). Below are the details of the dataset. (Codes used to preprocess the text can be found at functions/tp_func)

| #Label | #Num of Data | #Average Length |
|--------|--------------|-----------------|
| ESG    | 1,387        | 16,879          |
| NEWS   | 1,500        | 917             |

## Functions

The functions include implementations of readability assessment scores (FOG, RSRS). It also includes novel readability assessment scores introduced from our paper: sentimentAssessment and biRSRS. Both scores leverage pretrained language models, and is  designed to function in a zero-shot manner, not requiring additional labeld data or training of any sort. 


## Contributors

[Guijin Son](https://github.com/guijinSON),
[Naeun Yoon](https://github.com/shineeun),
[Kaeun Lee](https://github.com/Kaeun-Lee)

If you have any questions please feel free to reach out at [spthsrbwls123@yonsei.ac.kr](spthsrbwls123@yonsei.ac.kr).
