import csv, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from embeddings import *
from data_loading import *
from constants import *
from utils import *

def format_submission(labels):
    # returns a df ready to input to submission_to_csv
    labels = [y for x in labels for y in x]
    pred_df= pd.DataFrame({'Prediction': labels})
    pred_df.index.name = 'Id' # rename id column
    pred_df.index += 1 #shift to correspond to sample submission
    pred_df['Prediction'] = pred_df['Prediction'].replace("0","-1").replace(0,-1)
    return pred_df

def submission_to_csv(predictions, filename):
    if not os.path.exists(PREDS_FOLDER):
        os.makedirs(PREDS_FOLDER)
    predictions.to_csv(PREDS_FOLDER + filename + ".csv", index_label="Id")
    
def classify_bow(train, tweets_col = 'clean_tweet'):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(train[tweets_col])
    
    train_bow = bow

    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

    lreg = LogisticRegression()
    lreg.fit(xtrain_bow, ytrain) # training the model

    prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
    prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(np.int)

    print(prediction_int)
    print(f1_score(yvalid, prediction_int)) # calculating f1 score
    
def classify_tfidf(train, tweets_col = 'clean_tweet'):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(train[tweets_col])
    
    train_tfidf = tfidf
    
    # splitting data into training and validation set
    xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42, test_size=0.3)

    lreg = LogisticRegression()
    lreg.fit(xtrain_tfidf, ytrain)

    prediction = lreg.predict_proba(xvalid_tfidf)
    prediction_int = prediction[:,1] >= 0.3
    prediction_int = prediction_int.astype(np.int)
    print(prediction_int)
    print(f1_score(yvalid, prediction_int))
    
def classify_fasttext(train, tweets_col = "clean_tweet", csv_name = "fasttext_clean", max_iter = 200):
    best_precision = 0;
    i_best = -1;

    for i in range(0,max_iter):
        train_name, test_name = create_labeled_csv(train, tweets_col = "clean_tweet", split_test = True)
        classifier = fasttext.supervised(train_name, 'model_supervised', label_prefix='__label__')

        result = classifier.test(test_name)
        #print('P@1:', result.precision)
        #print('R@1:', result.recall)
        #print('Number of examples:', result.nexamples)
        if result.precision > best_precision:
            # print('\tprecision: ' + str(best_precision) + ' --> ' + str(result.precision))
            best_precision = result.precision
            i_best = i

            labels = classifier.predict(load_txt(TEST_DATA))
            submission_to_csv(format_submission(labels), csv_name)
            
        printOver('\033[1mclassifying:\033[0m '+ str( (i+1)/ max_iter * 100) + '%, best_acc=' + str(best_precision))