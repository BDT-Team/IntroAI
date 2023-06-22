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
    neutral = 0
    spam = 0
    sentences = [pipeline_preprocess(s) for s in raw_sentences]
    for s in sentences:
        if len(s) <= 0:
            sentences.remove(s)
            neutral += 1
            spam += 1
    sentences_CV = sentiment_tfidf.transform(sentences)
    pred_sentiment = sentiment_model.predict(sentences_CV.toarray())
    pred_sentiment_dict = Counter(pred_sentiment)
    
    sentences_CV = spam_tfidf.transform(sentences)
    pred_spam = spam_model.predict(sentences_CV.toarray())
    pred_spam_dict = Counter(pred_spam)
    
    return {"positive": pred_sentiment_dict[2],
            "neutral": pred_sentiment_dict[1]+neutral,
            "negative": pred_sentiment_dict[0],
            "spam": pred_spam_dict[1]+spam}