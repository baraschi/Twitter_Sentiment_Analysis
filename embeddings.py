import fasttext

def create_labeled_csv(tweets, tweets_col = "clean_tweet", label_col = "label", filename = "fasttext_labeled", split_test = False):
    # create column with correct label format for fasttext: '__label__0 '
    tweets['label_prefixed'] = tweets[label_col].apply(lambda s: '__label__' + str(s) + ' ')
    
    if(split_test):
        length = tweets.shape[0]
        test_name = "data/" + filename + "_test.txt"
        train_name = "data/" + filename + "_train.txt"
        # test set
        tweets.loc[:length/6].to_csv(test_name, columns = ['label_prefixed',tweets_col], index=False)
        #train set
        tweets.loc[length/6+ 1:].to_csv(train_name, columns = ['label_prefixed',tweets_col], index=False)
        return train_name, test_name
    else:
        name = "data/" + filename + ".txt"
        #generate csv file
        tweets.to_csv(name, columns = ['label_prefixed',tweets_col], index=False)
        return name
    
def fasttext_model(tweets, model = "skipgram", tweets_col = "clean_tweet", label_col = "label"):
    create_labeled_csv(tweets,tweets_col, label_col)
    # Skipgram model
    if model == "skipgram":
        model = fasttext.skipgram("data/labels_train.txt", 'model_skipgram')
        #print(model.words) # list of words in dictionary
    elif model == "cbow":
        model = fasttext.cbow("data/labels_train.txt", 'model_cbow')
        #print(model.words) # list of words in dictionary

    return model
