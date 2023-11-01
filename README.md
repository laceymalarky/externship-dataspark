# DataSpeak | Generative Q&A Large Language Model
## Project Overview
DataSpeak, one of the industry's largest providers of predictive analytics solutions, needed help developing a proof-of-concept machine learning model that can automatically generate answers to user-input questions.

## Machine Learning Skills/Technologies
Text2TextGeneration, Transformers, Tokenizers, PyTorch, Hugging Face, FLAN-T5 LLM, Word2Vec, GPU 

## Project Conclusions
- Developed a [generative fined-tuned language model](https://huggingface.co/lmalarky/flan-t5-base-finetuned-python_qa) using `google/flan-t5-base`, trained on a Stack Overflow dataset.
- Performed cosine semantic similarity analysis on a generated vector database to match the user-input question to top 5-most similar questions in the dataset.
- Built and a web application with a chatbot UI that takes a user question and returns a generative answer from the model and generates 5 additional possible answers based on cosine similarity.
- The model achieved a 19% ROUGE-1 score and an average perplexity of 1.96.
- The model is highly efficient, able to produce an answer in <10 seconds.
- Pre-processed raw data to improve the quality of the training set.

  

<img width="630" alt="Screenshot 2023-10-30 at 8 45 15 PM" src="https://github.com/laceymalarky/nlp_question_answer/assets/97048468/bc380430-48af-44fb-969f-198ba69053ba">
![Screenshot 2023-11-01 at 4 16 18 PM](https://github.com/laceymalarky/nlp_question_answer/assets/97048468/c94264f5-18cb-4ad3-8795-f8d64ec15a99)

## Requirements
Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, transformers, spacy, torch

## Data Description:
[Python Questions from Stack Overflow](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)
