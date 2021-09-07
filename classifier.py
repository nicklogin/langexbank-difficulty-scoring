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
from transformers import BertModel, BertTokenizer
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