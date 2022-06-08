# Wongnai-Review-Classifier
This repository contains the implementations of [BERT](https://arxiv.org/abs/1810.04805) and [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) models to classify the review ratings of the [Wongnai review dataset](https://github.com/wongnai/wongnai-corpus/blob/master/review/review_dataset.zip). The [Wongnai Challenge](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction) is also hosted on Kaggle, however, it appears to be only available to invited participants.

## Pre-processing
The review data is pre-processed using the [pythainlp](https://github.com/PyThaiNLP/pythainlp). In particular, the review consists largely of Thai but some of it is mixed with a little English. To process this mixture of Thai and English, the `process_thai` function within the `pythainlp` module is used to generate the word tokens of the training corpus. To filter out noise, only tokens which appear 10 times or more are included in the training vocabulary.

## Classifier Models
The classifier models implemented include (i) BERT, (ii) GPT and (iii) a customized BERT-Downsampled model which applies 1D average pooling to reduce the sequence length before sending it into the self-attention mechanism. The Masked Language Modeling (MLM) for this downsampled model is modified to predict the masked token within each sub-sequence that is pooled into the same kernel to be averaged across. 

In addition, the MLM for (i) is modified to let the CLS token output embedding mirror the average output embedding of the rest of the sequence since there is no Next Sentence Prediction (NSP) pre-training done. For the GPT model, the next word prediction is used to pre-train the model. For all 3 models, pre-training is done only on the training corpus and the classifier is a fully-connected layer whose input is the averaged output embeddings of the deep learning model applied. Mix-up augmentation is also applied to attempt to improve the models' performance, however, this enhancement did not appear to improve the models' accuracy. 

Note 1: For the downsampled BERT model, the kernel size is set to 3 regardless of the input sequence length.
Note 2: Due to computational constraints, all the models implemented were very small and had a depth of 3 layers, with 4 attention heads, 256 hidden units and 1024 feed-forward units. 

## Model Performance on Validation Dataset
The full dataset of 40,000 reviews is divided into 90% training data and 10% validation data. The model's performance is evaluated on the validation data, which is 4,000 reviews.

| Model | Seq. Length | Accuracy | Weighted Average F1-Score |
| ----- | ----------- | -------- | ------------------------- |
| BERT | 256 | 0.552 | 0.51 |
| GPT | 256 | 0.554 | 0.54 |
| BERT Downsampled | 300 | 0.574 | 0.55 |

It can be observed that the BERT Downsampled model outperformed the BERT and GPT counterparts, although it should also be taken into account that the input sequence length of the downsampled BERT model is longer than that of BERT and GPT models. Running the downsampled BERT model using the input sequence length of 256 is currently being done for a fair comparison and will be updated again.

## Compute Resources
The codes in this repository was run using an Intel i5 CPU processor without any GPU. The BERT and GPT models took around 20 hours to train, while the BERT Downsampled model took around 10 hours to train.

## Running the Scripts
To train the model, first download the [data](https://github.com/wongnai/wongnai-corpus/blob/master/review/review_dataset.zip) and process the corpus via
```
python process_thai_reviews_words.py
```
followed by
```
python train_wongnai_reviews_bert_pretrain.py
```
to train the BERT model, or
```
python train_wongnai_reviews_gpt_cls.py
```
to train the GPT model, or
```
python train_bert_wongnai_reviews_downsampled.py
```
to train the BERT Downsampled model.
