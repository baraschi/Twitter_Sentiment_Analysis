from constants import *
from data_loading import *
from data_cleaning import *
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():

    # load training set as DataFrame
    train = load_df(TRAIN_NEG, TRAIN_POS, TWEET, LABEL, LABEL_NEG, LABEL_POS)
    # set patterns to remove, replace, and replace with
    to_remove = "<user>"
    to_replace = "[^a-zA-Z#]"
    replace_value = " "
    # clean training set
    train = clean(train, TWEET, CLEAN_TWEET, to_remove, to_replace, replace_value)


if __name__ == '__main__':
    main()
