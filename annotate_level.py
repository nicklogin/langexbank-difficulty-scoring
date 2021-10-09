import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

from handlers import RuleHandler

def indicate_handler(row, handler_list):
    non_zeros = row.nonzero()[0]
    if not len(non_zeros):
        return "Supervised classifier"
    elif len(non_zeros) == 1:
        return handler_list[non_zeros[0]].__name__
    else:
        return "Error:> 1 Rule worked!"

def annotate_level(questions_df: pd.DataFrame, indicate_handlers=False):
    MODEL = spacy.load("en_core_web_sm")
    handler_results = []
    subclx = RuleHandler.__subclasses__()
    for handler_cls in subclx:
        handler = handler_cls(MODEL)
        
        tqdm.pandas()
        handler_result = questions_df.progress_apply(lambda x: handler(x['Sentence'],
                                                                       x['Error span'],
                                                                       x['Right answer'],
                                                                       x['Error type']),
                                                    axis=1)
        handler_results.append(handler_result)
    handler_results = np.array(handler_results)
    if indicate_handlers:
        triggered_handlers = np.apply_along_axis(lambda x: indicate_handler(x, subclx),
                                                 0, handler_results)
        return sum(handler_results), triggered_handlers
    return sum(handler_results)
