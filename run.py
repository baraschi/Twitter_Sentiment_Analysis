from constants import *
from data_loading import *
from data_cleaning import *
from utils import *
from prediction import *
import matplotlib.pyplot as plt
import itertools
import sys

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

def reproduce_best(best_row, train, test):
    classifier = None
    if(best_row.iloc[0]['method'] == 'bow'):
        classifier = classify_bow
    elif(best_row.iloc[0]['method'] == 'tfidf'):
        classifier = classify_tfidf

    clean_options = {
        'duplicates': best_row.iloc[0]['duplicates'],
        'replace_pattern': best_row.iloc[0]['replace_pattern'],
        'stop_words': best_row.iloc[0]['stop_words'],
        'stemming': best_row.iloc[0]['stemming'],
    }
    # set patterns to remove, replace, and replace with
    to_remove = "<user>"
    to_replace = "[^a-zA-Z#]"
    replace_value = " "

    print("Cleaning train...")
    train_new = clean(train, TWEET, CLEAN_TWEET, to_remove, to_replace, replace_value,clean_options)

    print("Cleaning test...")
    test_new = clean(test, TWEET, CLEAN_TWEET, to_remove, to_replace, replace_value,clean_options)


    print("Classifying...")
    accuracy = classifier(
        train_new,
        test_new,
        tweets_col = CLEAN_TWEET,
        ngram_range=(1,best_row.iloc[0]['n-gram']),
        max_features = best_row.iloc[0]['nb_features'],
        filename = "submission"
    )
    print("Created submission_clean_tweet.csv, with best train acc=",accuracy)

def main(full = "full"):
    # load training set as DataFrame
    print("Loading data...")

    train_pos_path = TRAIN_POS_FULL
    train_neg_path = TRAIN_NEG_FULL

    if full == "light":
        print("(with light dataset)")
        train_pos_path = TRAIN_POS
        train_neg_path = TRAIN_NEG

    train = load_df(train_neg_path, train_pos_path, TWEET, LABEL, LABEL_NEG, LABEL_POS)
    train.dropna(inplace=True)
    train.reset_index(drop=True,inplace=True)

    test = pd.DataFrame({TWEET: load_txt(TEST_DATA)})
    test.dropna(inplace=True)
    test.reset_index(drop=True,inplace=True)

    best_params = pd.DataFrame(columns=['method','n-gram','duplicates', 'replace_pattern','stop_words','stemming','nb_features','accuracy'])
    best_params = best_params.append({
        'method': 'tfidf',
        'n-gram': 3,
        'duplicates': 0,
        'replace_pattern': 1,
        'stop_words': 0,
        'stemming': 1,
        'nb_features': 130000
    }, ignore_index = True)

    reproduce_best(best_params, train, test)
    
if __name__ == '__main__':
    full = "full"
    if len(sys.argv) >= 2:
        full = sys.argv[1]
    main(full = full)
