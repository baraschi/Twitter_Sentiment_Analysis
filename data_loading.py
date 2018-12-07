import pandas as pd


# Load txt file
def load_txt(file):
    return open(file).readlines()


# Create DataFrame from loaded txt file.
# Set first column to data set content.
# Set second column to label value.
def create_df(dataset, col_1_name, col_2_name, label):
    df = pd.DataFrame()
    df[col_1_name] = dataset
    df[col_2_name] = label
    return df


# Merge two DataFrames together and reset id
def merge_df(df_1, df_2):
    df = pd.concat([df_1, df_2], ignore_index=True)
    df.index.name = 'id'
    return df


# Load positive and negative files, create DataFrame, set labels, merge together
def load_df(neg_file, pos_file, col_name, label_name, neg_label, pos_label):
    train_neg = load_txt(neg_file)
    # load positive set
    train_pos = load_txt(pos_file)
    # create negative DataFrame
    train_neg_df = create_df(train_neg, col_name, label_name, neg_label)
    # create positive DataFrame
    train_pos_df = create_df(train_pos, col_name, label_name, pos_label)
    # merge negative and positive DataFrames
    return merge_df(train_neg_df, train_pos_df)
