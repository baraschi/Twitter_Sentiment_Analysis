from constants import *
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
import re


def clean(data, old_column, new_column, pattern_to_remove, pattern_to_replace, replacement, options):
    if options['duplicates']:
        data.drop_duplicates(subset = [old_column], inplace = True)
        data = data.reset_index(drop=True)

    # remove pattern '<user>' from tweets, since it is useless for our analysis
    data[new_column] = np.vectorize(remove_pattern)(data[old_column], pattern_to_remove, '')
    # replace special characters, numbers and punctuations, except chars & hashtags

    if options['replace_pattern'] :
        data[new_column] = replace_pattern(data[new_column], pattern_to_replace, replacement)
        
    if options['stop_words'] :
    # remove stop words
        data[new_column] = remove_stop_words(data[new_column])

    if options['stemming'] :
        # tokenize data
        tokens = tokenize(data[new_column])
        # stem data
        stemmed_tokens = stem(tokens)
        # stitch stems back together
        data[new_column] = stitch(stemmed_tokens)
        
    return data


# find pattern in text and replace
def remove_pattern(text, pattern, replace):
    matches = re.findall(pattern, text)
    for match in matches:
        text = re.sub(match, replace, text)
    return text


# replace pattern in given column
def replace_pattern(column, pattern, replace):
    column = column.str.replace(pattern, replace)
    return column


# tokenize rows
def tokenize(column):
    return column.apply(lambda x: x.split())


# stem tokens
def stem(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])
    return stemmed_tokens


# stitch tokens back together
def stitch(tokens):
    for i in range(len(tokens)):
        tokens[i] = ' '.join(tokens[i])
    return tokens


# remove stop words
def remove_stop_words(column):
    stop_words = set(stopwords.words('english'))
    return column.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
