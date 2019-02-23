from collections import Counter

import numpy as np
from scipy import spatial


def normal_spearman(tuple_list, N):
    return 1 - ((6 * sum([(rank_a - rank_b) ** 2
                          for rank_a, rank_b in tuple_list])) / float(N ** 3 - N))


def word_similarity(model, word1, word2, tokenizer, use_unknown_vocab=True):
    v1_vecs = [model[word] for word in tokenizer.tokenize(word1) if word in model.vocab]
    v2_vecs = [model[word] for word in tokenizer.tokenize(word2) if word in model.vocab]
    if use_unknown_vocab:
        if v1_vecs and v2_vecs:
            v1 = np.mean(v1_vecs, axis=0)
            v2 = np.mean(v2_vecs, axis=0)
            return 1 - spatial.distance.cosine(v1, v2)
        else:
            return -1
    else:
        v1 = np.mean(v1_vecs, axis=0)
        v2 = np.mean(v2_vecs, axis=0)
        return 1 - spatial.distance.cosine(v1, v2)
