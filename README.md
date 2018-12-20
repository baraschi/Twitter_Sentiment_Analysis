------------------------------------------------------------------
Check this
------------------------------------------------------------------

– Reproducibility: In your submission, you must provide a script run.py which produces exactly the same .csv predictions which you used in your best submission to the competition on CrowdAI. This includes a clear ReadMe file explaining how to reproduce your setup, including precise training, prediction and installation instructions if additional libraries are used - the same way as if you would ideally instruct a new teammate to start working with your code.
– Documentation: Your ML system must be clearly described in your PDF report and also well- documented in the code itself. A clear ReadMe file must be provided. The documentation must also include all data preparation, feature generation as well as cross-validation steps that you have used.
– External ML libraries are allowed, as long as accurately cited and documented.
– External datasets are allowed, as long as accurately cited and documented.

------------------------------------------------------------------

# Twitter Sentiment Analysis

In this report, we present a study of sentiment analysis on Twitter data, where the task is to predict whether the smiley contained in the tweet is happy :) or sad :(. 
We experimented with today's most common solutions, such as text preprocessing and supervised classification techniques. We mixed-and-matched our algorithms to evaluate how it influenced the accuracy of our predictions. 
Our predictor currently obtains an accuracy of: ...  : 

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
 
 * [Matplotlib](https://matplotlib.org/)
    ```sh
    $ pip install matplotlib
    ```
### Files
* Train and Test Data

    Download all files [here](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files) in order to train and test the models
    and move them in `data/twitter-datasets/` directory.

* x.py
#### Functions
#### Function arguments
#### Return type

### Reproduce Our Submission
In order to produce the same submission corresponding to our crowdAI ranking, just run the following command:
```sh 
$ python3 run.py
```

The submission can be found in the file 'preds/submission_clean_tweet.csv'
### Contributors

- Zoé Baraschi
- Arnaud Boissaye
- Louis-Maxence Garret


   [Anaconda3]: <https://www.continuum.io/downloads>
   [Scikit-Learn]: <http://scikit-learn.org/stable/install.html>
   [NLTK]: <http://www.nltk.org/>
___

License: [MIT](https://opensource.org/licenses/MIT)
