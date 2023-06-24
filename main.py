import os

from typing import Union, List
from fastapi import FastAPI

import pickle, joblib
from collections import Counter
from data.preprocess import pipeline_preprocess

ALL_MODEL = "model"
SENTIMENT_FOLDER = "level0_sentiment"
SPAM_FOLDER = "level0_spam"

sentiment_tfidf = pickle.load(open(os.path.join(ALL_MODEL, SENTIMENT_FOLDER, "sentiment_tfidf.pkl"), 'rb'))
sentiment_model = joblib.load(os.path.join(ALL_MODEL, SENTIMENT_FOLDER, "nb.sav"))

spam_tfidf = pickle.load(open(os.path.join(ALL_MODEL, SPAM_FOLDER, "spam_tfidf.pkl"), 'rb'))
spam_model = joblib.load(os.path.join(ALL_MODEL, SPAM_FOLDER, "forest.sav"))

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/items")
async def read_item(raw_sentences: List[str] = None):
    null_sentence = 0
    raw_sentences_without_null = []
    for s in raw_sentences:
        if len(s.strip()) > 0:
            raw_sentences_without_null.append(s)
        else:
            null_sentence += 1
    sentences = [pipeline_preprocess(s) for s in raw_sentences_without_null]
    len_sentence_before = len(sentences)
    sentences = [s for s in sentences if len(s) > 0]
    no_meaning = len_sentence_before - len(sentences)
    
    sentences_CV = spam_tfidf.transform(sentences)
    pred_spam = spam_model.predict(sentences_CV.toarray())
    pred_spam_dict = Counter(pred_spam)
    
    sentences_CV = sentiment_tfidf.transform(sentences)
    pred_sentiment = sentiment_model.predict(sentences_CV.toarray())
    pred_sentiment_dict = Counter(pred_sentiment)
    
    pred_sentiment_ham = []
    for i in range(len(sentences)):
        if pred_spam[i] == 0:
            pred_sentiment_ham.append(pred_sentiment[i])
            
    pred_sentiment_ham_dict = Counter(pred_sentiment_ham)
    
    return {"all": len(raw_sentences),
            "all_without_null": len(raw_sentences_without_null),
            "all_ham_without_null": len(pred_sentiment_ham),
            "positive": pred_sentiment_dict[2],
            "neutral": pred_sentiment_dict[1] + no_meaning + null_sentence,
            "neutral_without_null": pred_sentiment_dict[1] + no_meaning,
            "negative": pred_sentiment_dict[0],
            "spam": pred_spam_dict[1] + no_meaning + null_sentence,
            "spam_without_null": pred_spam_dict[1] + no_meaning,
            "positive_ham": pred_sentiment_ham_dict[2],
            "neutral_ham": pred_sentiment_ham_dict[1] + null_sentence,
            "neutral_ham_without_null": pred_sentiment_ham_dict[1],
            "negative_ham": pred_sentiment_ham_dict[0]}