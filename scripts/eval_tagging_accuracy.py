from operator import itemgetter
import re
import sys
from typing import NamedTuple

import numpy as np

class POSsed(NamedTuple):
    idx: int
    word: str
    lemma: str
    pos: str
    mtag: str

# TODO: Make this able to evaluate accuracy of gzipped files as well
def output_generator(f_name, idx_col, form_col, lem_col, pos_col, mtag_col):
    with open(f_name, 'r') as to_f:
        indices_getter = itemgetter(idx_col, form_col, lem_col, pos_col, mtag_col)
        for line in filter(lambda line: not line.startswith('#') and line.strip(), to_f):
            row = line.strip().split('\t')
            try:
                idx, word, lemma, pos, mtag = indices_getter(row)
                yield POSsed(idx=idx, word=word, lemma=lemma, pos=pos, mtag=mtag)
            except IndexError as e:
                print(row, file=sys.stderr)
                raise e

EMPTY_ENTRY_ = re.compile(r'^_(?:;_)*$')

def tagging_accuracy(oracle_fname, pred_fname, train_fname, tag_to_str, idx_col, form_col, lem_col, pos_col, mtag_col, only_oov=False):
    conllu_open = lambda f_name: output_generator(f_name, idx_col, form_col, lem_col, pos_col, mtag_col)

    oracle = np.array([tag_to_str(p) for p in conllu_open(oracle_fname)])
    pred = np.array([tag_to_str(p) for p in conllu_open(pred_fname)])
    not_empty_masker = np.vectorize(lambda p: not EMPTY_ENTRY_.match(p))
    criterion = not_empty_masker(oracle)
    if only_oov:
        vocab = {p.word for p in conllu_open(train_fname)}
        isOOV = np.array([p.word not in vocab for p in conllu_open(oracle_fname)])
        criterion = criterion & isOOV
    return (pred == oracle)[criterion].mean()
