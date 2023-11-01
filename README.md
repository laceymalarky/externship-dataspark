# DataSpeak | Generative Q&A Large Language Model
## Project Overview
DataSpeak, one of the industry's largest providers of predictive analytics solutions, needed help developing a proof-of-concept machine learning model that can automatically generate answers to user-input questions.

## Machine Learning Skills/Technologies
Text2TextGeneration, Transformers, Tokenizers, PyTorch, Hugging Face, FLAN-T5 LLM, GPU 

## Project Conclusions
- Developed a generative fined-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) large language model on the [Python Questions](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) from Stack Overflow dataset which achives 19% ROUGE-1 score and average perplexity of 1.96.
- Performed time series analysis to reveal that the churn rate increased from 8% to 47% over the past 6 years with monthly seasonal swings of 9%.
- Analyzed and pre-processed raw data, applied feature encoding and engineering to prepare features for modeling.

  https://huggingface.co/lmalarky/flan-t5-base-finetuned-python_qa
  
## Requirements
Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, transformers, spacy, torch

## Data Description:
https://www.kaggle.com/datasets/stackoverflow/pythonquestions
