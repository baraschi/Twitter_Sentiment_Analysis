import fasttext

def fasttext_model(tweets, model = "skipgram", tweets_col = "clean_tweet_no_stop", label_col = "label"):
    # create column with correct label format for fasttext: '__label__0 '
    tweets['label_prefixed'] = tweets['label'].apply(lambda s: '__label__' + str(s) + ' ')
    
    #generate csv file
    tweets.to_csv("data/fasttext_train.txt", columns = ['label_prefixed','clean_tweet_no_stop'], index=False)
    
    # Skipgram model
    if model == "skipgram":
        model = fasttext.skipgram("data/labels_train.txt", 'model_skipgram')
        #print(model.words) # list of words in dictionary
    elif model == "cbow":
        model = fasttext.cbow("data/labels_train.txt", 'model_cbow')
        #print(model.words) # list of words in dictionary

    return model
