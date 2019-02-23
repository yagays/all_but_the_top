import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors


def all_but_the_top(X, D=3):
    """
    ref. https://github.com/woctezuma/steam-descriptions/blob/master/sif_embedding_perso.py
    """
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=D)
    pca.fit(X_centered)

    U = pca.components_
    X_prime = X_centered - X_centered.dot(U.transpose()).dot(U)
    return X_prime


if __name__ == "__main__":
    # word2vec
    w2v_d = [100, 200, 300]
    for num_d in w2v_d:
        print(f"Processing: {num_d}")
        w2v_path = f"/Users/okuda/dev/nlp/pretrained_embedding/WikiEntVec/jawiki.word_vectors.{num_d}d.txt"
        model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

        D = int(num_d / 100)
        output_vec = all_but_the_top(model.vectors, D=D)
        # output
        model.vectors = output_vec
        model.save_word2vec_format(f"model/jawiki.word_vectors.{num_d}d.abtt.bin", binary=True)

    # fasttext
    print(f"Processing: fasttext300d")
    w2v_path = f"/Users/okuda/dev/nlp/pretrained_embedding/fasttext/cc.ja.300.vec"
    model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

    D = 3
    output_vec = all_but_the_top(model.vectors, D=D)
    # output
    model.vectors = output_vec
    model.save_word2vec_format(f"model/cc.ja.300d.abtt.bin", binary=True)
