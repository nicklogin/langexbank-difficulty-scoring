import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

import base

MODEL = spacy.load("en_core_web_sm")

def annotate_level(questions_df: pd.DataFrame):
    handler_results = []
    for handler_cls in base.RuleHandler.__subclasses__():
        handler = handler_cls(MODEL)
        
        tqdm.pandas()
        handler_result = questions_df.progress_apply(lambda x: handler(x['Sentence'],
                                                                       x['Error span'],
                                                                       x['Right answer'],
                                                                       x['Error type']),
                                                    axis=1)
        handler_results.append(handler_result)
    handler_results = np.array(handler_results)
    return sum(handler_results)
