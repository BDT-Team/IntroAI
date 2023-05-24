# IntroAI

Labelling rules:
- Cột spam:
  + Nếu spam: điền 1
  + Nếu không spam: điền 0
- Cột sentiment:
  + Tích cực: điền 1
  + Trung tính: điền 0
  + Tiêu cực: điền -1

## Progress (updated 24/5/2023)
- Data crawling: 
  + Technology: Python BeautifulSoup
  + ~14000 comments in 10 Shopee posts with different product types. 
  + I also use [this dataset from Kaggle](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst) with ~31000 comments
- Data labeling: manually done
  + Each comment has 2 labels: Spam and Sentiment
  + Labelling rules: mentioned above
- Data preprocessing: Done
  + Technology: Python (numpy, regex, underthesea
  + Operation: remove emojies, punctuation, stopwords, repeated last characters (haaaa->ha); replace phone number, datetime, link; word segmentation and text normalization
- Build ML models from scratch: Done
  + Technology: Python (numpy)
  + Model list: KNN, Multiclass Logistic Regression (Softmax Regression), Multiclass SVM, Decision Tree, Random Forest, Multinomial Naive Bayes
  + Note: I used Gini Impurity instead of Entropy in Decision Tree. I also implemented Adam optimizer in Multiclass SVM
- Build sentiment analysis models:
  + Technique: Stratified k-fold cross validation, SMOTE, SMOTE+Tomeklink
  + Train-test split: 0.8 - 0.2, then test dataset is discard randomly so that each class has equal number of comments.
  + Dataset: [this dataset from Kaggle](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst)
  + Result (based on accuracy)
    |  |KNN|Softmax|SVM|Tree|RF|MultinomialNB|
    |---|---|---|---|---|---|---|
    |Normal|0.476|0.565|0.393|0.556|0.566|0.545|
    |SMOTE|0.521|0.369|0.529|0.547|0.58|0.612|
    |CV|0.476|0.562|0.391|0.545|0.564|0.545|
    |SMOTETomek|0.527|0.370|0.538|0.544|0.604|0.625|


