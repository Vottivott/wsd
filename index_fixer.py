import numpy as np
import os


def load_embedding_batch(index, name, dataset_name):
    emb_path = "embeddings/" + name + "/" + dataset_name
    labels_path = emb_path + " labels"
    lemmas_path = emb_path + " lemmas"
    emb = np.load(emb_path + "/" + str(index) + ".npy")
    labels = np.load(labels_path + "/" + str(index) + ".npy")
    lemmas = np.load(lemmas_path + "/" + str(index) + ".npy")
    return emb, labels, lemmas


def get_num_batches(name, dataset_name):
    emb_path = "embeddings/" + name + "/" + dataset_name
    return len(os.listdir(emb_path))


def get_embedding_size(name, dataset_name):
    emb_path = "embeddings/" + name + "/" + dataset_name
    emb = np.load(emb_path + "/" + str(0) + ".npy")
    return emb.shape[1]


jobs = [(f,f.replace("fix ","")) for f in os.listdir("embeddings") if f.startswith("fix ")]
for folder, newpath in jobs:
    newpath = "embeddings/" + newpath
    ds = ['trn', 'vld']
    for d in ds:
        n = np.load(d + '_indices.npy')
        o = np.load(d + '_indices_old.npy')
        emb_rows = []
        labels_list = []
        lemmas_list = []
        for i in n:
            print('.',end="")
            old_i = np.where(o == i) # the position in old of index, i.e. where to fetch this row
            file_index = int(old_i / 32)
            file_row = old_i % 32
            emb, labels, lemmas = load_embedding_batch(file_index, folder, d)
            emb_rows.append(emb[file_row,:])
            labels_list.append(labels[file_row])
            lemmas_list.append(lemmas[file_row])
            if i % 32 == 31:
                p = newpath + "/" + d
                suf = "/" + str(i) + ".npy"
                np.save(p+suf, np.vstack(emb_rows))
                np.save(p+" labels" + suf, labels_list)
                np.save(p+" lemmas" + suf, lemmas_list)
                emb_rows = []
                labels_list = []
                lemmas_list = []




