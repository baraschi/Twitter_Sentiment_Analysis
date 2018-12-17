import csv, os, collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from embeddings import *
from data_loading import *
from constants import *
from utils import *

def format_submission(labels):
    # returns a df ready to input to submission_to_csv
    
    if isinstance(labels[0], collections.Iterable):
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

def predict_and_submit(classifier, test_texts, filename):
    labels = classifier.predict(test_texts)
    submission_to_csv(format_submission(labels), filename)
    
def classify_bow(train, test, tweets_col = CLEAN_TWEET, filename = "bow", max_df = 0.90, min_df  = 1, max_features=1000, ngram_range=(1,1)):
    bow_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words='english', ngram_range=ngram_range)
    
    # bag-of-words feature matrix
    train_bow = bow_vectorizer.fit_transform(train[tweets_col])
    test_bow = bow_vectorizer.transform(test[tweets_col])
    
    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)
    lreg_validation = LogisticRegression()
    lreg_validation.fit(xtrain_bow, ytrain) # training the model
    prediction_validation = lreg_validation.predict(xvalid_bow) # predicting on the validation set
    accuracy = accuracy_score(prediction_validation,yvalid)
    
    # regression using test set
    lreg_test = LogisticRegression()
    lreg_test.fit(train_bow, train['label']) # training the model
    prediction_test = lreg_test.predict(test_bow)
    
    submission_to_csv(format_submission(prediction_test.tolist()), filename + "_" + tweets_col)
    
    return accuracy
    
def classify_tfidf(train, test, tweets_col = CLEAN_TWEET, filename = "tfidf", max_df = 0.90, min_df = 1, max_features=1000, ngram_range=(1,1)):
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words='english', ngram_range=ngram_range)

    # TF-IDF feature matrix
    train_tfidf = tfidf_vectorizer.fit_transform(train[tweets_col])
    test_tfidf = tfidf_vectorizer.transform(test[tweets_col])
   
    # splitting data into training and validation set
    xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42, test_size=0.3)

    lreg_validation = LogisticRegression()
    lreg_validation.fit(xtrain_tfidf, ytrain)
    #prediction_validation = lreg_validation.predict(xvalid_tfidf) # predicting on the validation set
    prediction_validation = lreg_validation.predict_proba(xvalid_tfidf) # predicting on the validation set
    prediction_validation_int = prediction_validation[:,1] >= 0.5 # if prediction is greater than or equal to 0.5 than 1 else 0
    prediction_validation_int = prediction_validation_int.astype(np.int)
    accuracy = accuracy_score(prediction_validation_int,yvalid)

    # regression using test set
    prediction_test = lreg_validation.predict_proba(test_tfidf) # predicting on the test set
    prediction_test_int = prediction_test[:,1] >= 0.5 # if prediction is greater than or equal to 0.5 than 1 else 0
    prediction_test_int = prediction_test_int.astype(np.int)
    
    submission_to_csv(format_submission(prediction_test_int.tolist()), filename + "_" + tweets_col)
    
    return accuracy
    
def classify_fasttext(train, test, tweets_col = "clean_tweet", filename = "fasttext", max_iter = 200):
    filename = "fasttext_labeled"
    best_accuracy = 0;
    i_best = -1;

    for i in range(0,max_iter):
               # create column with correct label format for fasttext: '__label__0 '
        train['label_prefixed'] = train['label'].apply(lambda s: '__label__' + str(s) + ' ')
        train_fasttext, validation_fasttext = train_test_split(train[['label_prefixed',tweets_col]], random_state=42, test_size=0.3)

        train_validation_name = "data/" + filename + "_train_validation.txt"

        #train set
        train_fasttext.to_csv(train_validation_name, columns = ['label_prefixed',tweets_col], index=False)

        classifier_validation = fasttext.supervised(train_validation_name, 'model_supervised', label_prefix='__label__')

        #here we append a ' ' char at the end to avoid an IndexOutOfBound exception
        labels_validation = classifier_validation.predict(validation_fasttext[tweets_col].apply(lambda s: str(s) + ' '))
        
        #formatting
        validation_fasttext['label'] = validation_fasttext['label_prefixed'].apply(lambda s: int(s.replace("__label__", "").strip()))
        labels_validation = [int(y) for x in labels_validation for y in x]
        
        accuracy = accuracy_score(validation_fasttext['label'].tolist(), labels_validation)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            i_best = i
            
            #we have a better result so we predict on test set
            train_name = "data/" + filename + "_train.txt"
            train[['label_prefixed',tweets_col]].to_csv(train_name, columns = ['label_prefixed',tweets_col], index=False)
            classifier_test = fasttext.supervised(train_name, 'model_supervised', label_prefix='__label__')
            labels_test = classifier_test.predict(test[tweets_col].apply(lambda s: str(s) + ' '))

            submission_to_csv(format_submission(labels_test), filename + "_" + tweets_col)

            #labels = classifier.predict(load_txt(TEST_DATA))
            #submission_to_csv(format_submission(labels), csv_name)



        printOver('\033[1mclassifying:\033[0m '+ str( (i+1)/ max_iter * 100) + '%, best_acc=' + str(best_accuracy))
    print("\n")
    return best_accuracy