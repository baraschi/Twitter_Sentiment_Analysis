import re


# function to collect hashtags
def extract_hashtags(text):
    hashtags = []
    # Loop over the words in the tweet
    for t in text:
        hashtag = re.findall(r"#(\w+)", t)
        hashtags.append(hashtag)
    return hashtags


# count hashtags
def count_hashtags(hashtags):
    return sum(hashtags, [])
