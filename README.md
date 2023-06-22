# IntroAI

## Experiment
- Run `uvicorn main:app --reload`

## Labelling rules:
- Cột spam:
  + Nếu spam: điền 1
  + Nếu không spam: điền 0
- Cột sentiment:
  + Tích cực: điền 1
  + Trung tính: điền 0
  + Tiêu cực: điền -1

## Progress (updated 22/6/2023)
- Data crawling: 
  + Technology: Python Selenium
  + ~14000 comments in 10 Shopee posts with different product types. 
  + I also use [this dataset from Kaggle](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst) with ~31000 comments
- Data labeling: manually done
  + Each comment has 2 labels: Spam and Sentiment
  + Labelling rules: mentioned above
- Data preprocessing: Done
  + Technology: Python (numpy, regex, underthesea)
  + Operation: remove emojies, punctuation, stopwords, repeated last characters (haaaa->ha); replace phone number, datetime, link; word segmentation and text normalization
- Build ML models from scratch: Done
  + Technology: Python (numpy)
  + Model list: KNN, Multiclass Logistic Regression (Softmax Regression), Multiclass SVM, Decision Tree, Random Forest, Multinomial Naive Bayes
  + Note: I used Gini Impurity instead of Entropy in Decision Tree. I also implemented Adam optimizer in Multiclass SVM
- Build sentiment analysis models:
  + Technique: SMOTE, SMOTE+Tomeklink
  + Train-test split: 0.8 - 0.2, then test dataset is discard randomly so that each class has equal number of comments.
  + Dataset: [this dataset from Kaggle](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst). 
  + TF-IDF is used for numerical representation of words.
  + Hyperparameters: go to sentiment_analysis.py for more detail.
  + Result (based on accuracy):
    |  |KNN|Softmax|SVM|Tree|RF|MultinomialNB|
    |---|---|---|---|---|---|---|
    |Normal|0.476|0.565|0.393|0.556|0.566|0.545|
    |SMOTE|0.521|0.369|0.529|0.547|0.58|0.612|
    |SMOTETomek|0.527|0.370|0.538|0.544|0.604|**0.625**|
- Build spam classification models:
  + Technique: SMOTE, SMOTE+Tomeklink
  + Train-test split: 0.8 - 0.2, then test dataset is discard randomly so that each class has equal number of comments.
  + Dataset: data crawled from Shopee, and manually be labeled in Khoa_comments.xlsx and Binh_comments.xlsx 
  + TF-IDF is used for numerical representation of words.
  + Hyperparameters: go to spam_classification.py for more detail.
  + Result (based on accuracy):
    |  |KNN|Softmax|SVM|Tree|RF|MultinomialNB|
    |---|---|---|---|---|---|---|
    |Normal|0.738|0.735|0.603|0.803|0.838|0.723|
    |SMOTE|0.707|0.692|0.697|0.809|**0.852**|0.783|
    |SMOTETomek|0.711|0.696|0.7|0.808|0.849|0.787|
- Note: I used sklearn to implement KNN, Decision Tree and Random Forest due to its speed and memory efficiency. The self-implemeted version still works fine.

## Future work
- Using stacking ensemble technique to create better model (hope so)
