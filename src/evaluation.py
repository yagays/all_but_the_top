from pathlib import Path
from collections import Counter

import MeCab
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from metric import normal_spearman, word_similarity


class MeCabTokenizer():
    def __init__(self, mecab_args=""):
        self.tagger = MeCab.Tagger(mecab_args)
        self.tagger.parse("")

    def tokenize(self, text):
        return self.tagger.parse(text).strip().split()


tokenizer = MeCabTokenizer("-O wakati")
use_unknown_vocab = True

corpus_path = Path("/Users/okuda/dev/nlp/JapaneseWordSimilarityDataset/")

# for word2vec
# ===specify your word2vec model===
w2v_path = "/Users/okuda/dev/nlp/pretrained_embedding/WikiEntVec/jawiki.word_vectors.300d.txt"
model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

# for fasttext
# model = KeyedVectors.load_word2vec_format("/Users/okuda/dev/nlp/pretrained_embedding/fasttext/cc.ja.300.vec")
# model = KeyedVectors.load_word2vec_format("/Users/okuda/dev/nlp/all-but-the-top/model/cc.ja.300d.abtt.bin", binary=True)

for csv_path in corpus_path.glob("*.csv"):
    print(f"Processing... :{csv_path.stem}")

    df = pd.read_csv(csv_path)
    df["w2v_score"] = df.apply(lambda x: word_similarity(
        model, x["word1"], x["word2"], tokenizer, use_unknown_vocab), axis=1)
    if not use_unknown_vocab:
        df.dropna(inplace=True)

    # calculate the spearman rank correlation coefficient
    df["w2v_rank"] = df["w2v_score"].rank(method="min", ascending=False).astype(int)
    df["human_rank"] = df["mean"].rank(method="min", ascending=False).astype(int)

    rank_tuple_list = [tuple(x) for x in df[["w2v_rank", "human_rank"]].values]
    sp_coeff = normal_spearman(rank_tuple_list, len(rank_tuple_list)) * 100
    print(f"spearman: {sp_coeff}")

    df[["word1", "word2", "mean", "w2v_score"]].to_csv(f"data/{csv_path.stem}_result.csv", index=None)
