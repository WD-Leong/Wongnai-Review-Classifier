import os
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
from pythainlp.ulmfit import process_thai

# Load the training data. #
tmp_file = "../../Data/wongnai_reviews/w_review_train.csv"
tmp_data = pd.read_csv(tmp_file, sep=";")
tmp_cols = ["review", "rating"]
tmp_data.columns = tmp_cols
print(tmp_data.head())

# Split into training and validation datasets. #
rng = np.random.default_rng(12345)
idx = rng.permutation(len(tmp_data))
n_train = int(0.90 * len(tmp_data))

train_corpus = [
    tmp_data.iloc[x]["review"] for x in idx[:n_train]]
train_labels = [
    tmp_data.iloc[x]["rating"] for x in idx[:n_train]]
valid_corpus = [
    tmp_data.iloc[x]["review"] for x in idx[n_train:]]
valid_labels = [
    tmp_data.iloc[x]["rating"] for x in idx[n_train:]]

# Load the test data. #
test_file = "../../Data/wongnai_reviews/test_file.csv"
test_data = pd.read_csv(test_file, sep=";")
print(test_data.head())

train_data = []
w_counter  = Counter()
for n in range(len(train_corpus)):
    tmp_rating = train_labels[n]
    tmp_review = train_corpus[n]
    
    # Thai word tokenizer. #
    tmp_tokens = [
        x for x in process_thai(tmp_review) if x != ""]
    
    w_counter.update(tmp_tokens)
    train_data.append((tmp_rating, tmp_tokens))
    if (n+1) % 5000 == 0:
        print(n+1, "reviews out of", len(train_corpus), "processed.")

valid_data = []
for n in range(len(valid_corpus)):
    tmp_rating = valid_labels[n]
    tmp_review = valid_corpus[n]
    
    # Thai word tokenizer. #
    tmp_tokens = [
        x for x in process_thai(tmp_review) if x != ""]
    valid_data.append((tmp_rating, tmp_tokens))
    
    if (n+1) % 1000 == 0:
        print(n+1, "out of", len(valid_corpus), "reviews processed.")

test_sw_data = []
for n in range(len(test_data)):
    tmp_row = test_data.iloc[n]
    tmp_review = tmp_row["review"]
    
    # Thai word tokenizer. #
    tmp_tokens = [
        x for x in process_thai(tmp_review) if x != ""]
    test_sw_data.append(tmp_tokens)
    
    if (n+1) % 5000 == 0:
        print(n+1, "out of", len(test_data), "reviews processed.")

# Generate the subword vocabulary. #
min_count  = 10
word_vocab = list(sorted(
    [x for (x, y) in w_counter.items() if y >= min_count]))
idx_2_word = dict(
    [(x, word_vocab[x]) for x in range(len(word_vocab))])
word_2_idx = dict(
    [(word_vocab[x], x) for x in range(len(word_vocab))])
print("Vocab size:", len(word_vocab), "tokens.")

train_len = [len(y) for x, y in train_data]
valid_len = [len(y) for x, y in valid_data]
test_len  = [len(x) for x in test_sw_data]

print("95P Train Length:", np.quantile(train_len, 0.95))
print("95P Valid Length: ", np.quantile(valid_len, 0.95))
print("95P Test Length: ", np.quantile(test_len, 0.95))

tmp_path = "../../Data/wongnai_reviews/"
tmp_pkl_file = tmp_path + "thai_reviews_word.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(train_data, tmp_file_save)
    pkl.dump(valid_data, tmp_file_save)
    pkl.dump(test_sw_data, tmp_file_save)

    pkl.dump(word_vocab, tmp_file_save)
    pkl.dump(idx_2_word, tmp_file_save)
    pkl.dump(word_2_idx, tmp_file_save)
