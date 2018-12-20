# Twitter Sentiment Analysis

In this report, we present a study of sentiment analysis on Twitter data, where the task is to predict whether the smiley contained in the tweet is happy :) or sad :(. 
We experimented with today's most common solutions, such as text preprocessing and supervised classification techniques. We mixed-and-matched our algorithms to evaluate how it influenced the accuracy of our predictions. 
Our predictor currently obtains an accuracy of: 0.85

## Description

See [Project Description](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification)

## Dependencies

In order to run the project you will need the following dependencies installed:

### Libraries

* [Anaconda3](https://www.anaconda.com/download/) - Download and install Anaconda with python3
* [Scikit-Learn](https://scikit-learn.org/stable/) - Download scikit-learn library with conda

    ```sh
    $ conda install scikit-learn
    ```
    
* [Pandas](https://pandas.pydata.org/)

    ```sh
    $ conda install pandas
    ```
    
* [NLTK](https://www.nltk.org/) - Download all packages of NLTK

    ```sh
    $ python
    $ >>> import nltk
    $ >>> nltk.download()
    ```
 download all packages from the GUI
 
 * [Matplotlib](https://matplotlib.org/) - *Optional* - Needed to see the beautiful plots on our notebook!
    ```sh
    $ pip install matplotlib
    ```
### Files
* Train and Test Data

    Download all files [here](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files) in order to train and test the models
    and move them in `data/twitter-datasets/` directory.

* data_cleaning.py - methods used for data cleaning
* data_exploration.py - methods used to explore the data, like exctracting and countring hashtags
* data_loading.py - methods used for data loading and DataFrame creation
* prediction.py - methods to classify (BoW, TD-IDF), to cross-validate and to create the submission csv.
* run.py - main class, uses above functions to generate best available submission.
* Run_All_Combinations.ipynb - notebook we used to find best parameter combinations and to generate plots
* Data_Exploration.ipynb - notebook we used to explore the data to find out what cleaning methods we needed to apply

### Reproduce Our Submission
In order to produce the same submission corresponding to our [crowdAI](https://www.crowdai.org/challenges/epfl-ml-text-classification) ranking, just run the following command:
```sh 
$ python3 run.py
```

The submission can be found in the file __preds/submission_clean_tweet.csv__

Our submission can be found [here](https://www.crowdai.org/challenges/epfl-ml-text-classification) with username __baraschi__ and submission id __24870__
### Contributors

- Zoé Baraschi
- Arnaud Boissaye
- Louis-Maxence Garret
___

License: [MIT](https://opensource.org/licenses/MIT)
