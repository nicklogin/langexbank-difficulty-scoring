from annotate_level import annotate_level
from classifier import BERTEmbedder, predict_level
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score

import pandas as pd

import re
import pickle

def process_df(df: pd.DataFrame, indicate_handlers=False):
    with open("BertModel.pickle", 'rb') as inp:
        bert_model = pickle.load(inp)

    with open("BertTokenizer.pickle", 'rb') as inp:
        bert_tokenizer = pickle.load(inp)

    bert_embedder = BERTEmbedder(bert_model, bert_tokenizer)

    with open("TreeBERT.pickle", 'rb') as inp:
        bert_tree = pickle.load(inp)
    
    df['Error span'] = df['Sentence'].apply(lambda x:\
                                            re.search('<b>(.*?)</b>',
                                            x).group(1))
    if indicate_handlers:
        df["predicted"], df["Rule"] = annotate_level(df, indicate_handlers=True)
    else:
        df["predicted"] = annotate_level(df)
    df_zero = df[df["predicted"]==0]
    if len(df_zero):
        df_zero["predicted"] = predict_level(df_zero["Sentence"].to_list(), bert_embedder, bert_tree)
        df_zero["predicted"] = df_zero["predicted"].apply(lambda x: int(x.split("_")[1]))
        df.loc[df_zero.index, "predicted"] = df_zero["predicted"]
    return df
