# DataSpeak | Generative Q&A Large Language Model
## Project Overview
DataSpeak, one of the industry's largest providers of predictive analytics solutions, needed help developing a proof-of-concept machine learning model that can automatically generate answers to user-input questions.

## Machine Learning Skills/Technologies
Text2TextGeneration, Transformers, Tokenizers, PyTorch, Hugging Face, FLAN-T5 LLM, GPU 

## Project Conclusions
- Developed a [generative fined-tuned language model](https://huggingface.co/lmalarky/flan-t5-base-finetuned-python_qa) using `google/flan-t5-base`, trained on a Stack Overflow dataset.
- Achieved 19% ROUGE-1 score and an average perplexity of 1.96.
- Analyzed and pre-processed raw data, applied feature encoding and engineering to prepare features for modeling.

  

<img width="630" alt="Screenshot 2023-10-30 at 8 45 15 PM" src="https://github.com/laceymalarky/nlp_question_answer/assets/97048468/bc380430-48af-44fb-969f-198ba69053ba">

## Requirements
Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, transformers, spacy, torch

## Data Description:
[Python Questions from Stack Overflow](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)
