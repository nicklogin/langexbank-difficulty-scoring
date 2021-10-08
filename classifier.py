from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors
from tqdm import tqdm
from math import ceil

import re
import os
import html
import pandas as pd
import numpy as np

import nltk
from nltk import word_tokenize

def clean_question_text(qtext):
  quest_text = re.sub('<.*?>','',qtext)
  quest_text = html.unescape(quest_text)
  return quest_text

class MyBatchIterator:
  def __init__(self, texts, batch_size):
    self.texts = texts
    self.batch_size = batch_size
  
  def __iter__(self):
    self.start = 0
    return self
  
  def __next__(self):
    if self.start >= len(self.texts):
      raise StopIteration
    batch = self.texts[self.start:self.start+self.batch_size]
    self.start += self.batch_size
    return batch
  
  def __len__(self):
    return ceil(len(self.texts)/self.batch_size)

class BERTEmbedder:
  def __init__(self, model, tokenizer):
    self.tokenizer = tokenizer
    self.model = model
  
  def process(self, texts, flatten_method='average'):
    tokenized = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    ids = tokenized['input_ids']
    mask = tokenized['attention_mask']
    processed = self.model(input_ids=ids, attention_mask=mask)

    if flatten_method == 'average':
      return processed['last_hidden_state'].detach().numpy().mean(axis=1)
    elif flatten_method == 'pooler':
      return processed['pooler_output'].detach().numpy()
  
  def process_sample(self, texts, batch_size=4, flatten_method='average'):
    text_iter = MyBatchIterator(texts, batch_size=batch_size)
    batches = []

    for batch in tqdm(text_iter, total=len(text_iter)):
      batches.append(self.process(batch, flatten_method=flatten_method))
    
    return np.concatenate(batches, axis=0)

class GPT2Embedder(BERTEmbedder):
  def __init__(self, model, tokenizer):
    super().__init__(model, tokenizer)
    self.tokenizer.pad_token = self.tokenizer.eos_token
  
  def process(self, texts, flatten_method='average'):
    tokenized = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    ids = tokenized['input_ids']
    mask = tokenized['attention_mask']
    processed = self.model(input_ids=ids, attention_mask=mask)

    if flatten_method == 'average':
      return processed['last_hidden_state'].detach().numpy().mean(axis=1)
    elif flatten_method == 'pooler':
      return processed['pooler_output'].detach().numpy()

def predict_level(sentences, embedder, clf):
  X = embedder.process_sample(sentences)
  y_pred = clf.predict(X)
  return y_pred
