
"""
    Natural Language Processing -- Big Data Programming 1 Project
    Machine Learning model to predict the rating based on reviews.
    The code implements pipeline feature to evaluate the performance
    for multiple models.
"""
import nltk
import pandas as pd
import string, time
# do the import for stopwords
from nltk.corpus import stopwords
#Importing CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Importing some models to test
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
# imports for model evaluations
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
#Multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

"""
    Function to process the text reviews:
        Removing punctuations.
        joining the characters back to words.
        Removing stopwords.
    """
def process_text(raw_text):
    # Check for the punctuations 
    nopunc = [
              char for char in raw_text
              if char not in
              string.punctuation
              ]
              
    # Join the characters 
    nopunc = ''.join(nopunc)
    
    # Remove stopwords (if any)
    return [
            word for word in nopunc.split()
            if word.lower() not in
            stopwords.words('english')
            ]

"""
    Classifying rating to ‘Good’ and ‘Excellent’ based on stars:
        1, 2 & 3 stars ==> Good
        4 & 5 stars ==> Excellent
    """
def define_rating_class(rating):
    stars = [1,2,3]
    if rating in stars:
        return 'Good'
    else:
        return 'Excellent'

"""
    Model Evaluation:
        Printing confusion matrix and evaluation report.
        If needed, we can save the report as separate output file.
    """
def model_evaluation(label, y_test, pred):
    print (label)
    print (confusion_matrix(y_test, pred))
    print (classification_report(y_test, pred))

"""
    main() starts here
    """
def main():
    """
        Initially, we are reading .csv file, we need 'rating' and 'reviewText'.
        We can make changes to read directly from cassendra to read required
        columns.
        """
    df1 = pd.read_csv('data_clean.csv')
    df = df1[[
              'rating',
              'reviewText',
              'rev_len'
              ]]
    """
        NaN will raise TypeError - float.......!.
        Let's deal with the possible NaN. There are not mane NaN (less than 1% in reviews)
        We can't write reviews to fill in. So dropping all rows with NaN.
        """
    df=df.dropna()
    
    """
        Spliting rating to good (1 to 3 stars) and
        excellent (4 & 5 Stars).
        """
    df['rating']= df['rating'].apply(
                                     define_rating_class
                                     )
    """
        Separating data into feature ‘X’ and target ‘y’.
        Doing train test split for training and validating the model.
        """

    X = df['reviewText']
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y,
                                                        test_size=0.33,
                                                        random_state=42
                                                        )
    
    """ Tokenization:
        Let's apply process_text() to the 'reviewText' column in our dataframe to get the tokens.
        
        Vectorization:
        After Tokenization, in order to do Machine Learning, we need to convert each review into
        a vector form, so that the Machine Learning model can understand.
        CountVectorizer() is going to do the trick, which convert a collection of text documents
        to a matrix of token counts (This will be a Sparse Matrix). We can pass a range of parameters
        to the CountVectorizer. In this case, we are going to pass analyzer = process_text, which
        is our own created function. In the output, we will get a BoW after processing text according
        to the analyzer.
        We will do the following steps using Bag of Words (BoW) model now.
        Term Frequency, by counting how many times does a word appeared in each message.
        Inverse Document Frequency, which is actually a weigh the counts, frequent tokens get lower weight.
        Normalize the vectors to unit length, to abstract from the original text length (L2 norm).
        
        Pipeline:
        The purpose of the pipeline is to assemble several steps that can be cross-validated together while
        setting different parameters. We can set up all the transformation (given above) along with training
        the model in a single unit using pipeline feature of scikit-learn. Rather than doing all steps
        one-by-one, we can then call that single unit for our data processing. In this way, we save lots of
        time and there is no need to re-do all the transformation steps manually. A simple call of pipeline
        object, with stored steps, on the data will do all the processing in future. Further more, we can even
        create list of pipeline for several models for comparisons. Let’s do this all together.
        """
    
    pipelines = [
                 ('MultinomialNB_model', Pipeline([
                                                   ('bow', CountVectorizer(analyzer = process_text)),
                                                   ('tfidf', TfidfTransformer()),
                                                   ('model_nb', MultinomialNB())])),
                 ('LogisticRegression_model', Pipeline([
                                                        ('bow', CountVectorizer(analyzer = process_text)),
                                                        ('tfidf', TfidfTransformer()),
                                                        ('model_nb', LogisticRegression())])),
                 ('RandomForest_model', Pipeline([
                                                  ('bow', CountVectorizer(analyzer = process_text)),
                                                  ('tfidf', TfidfTransformer()),
                                                  ('model_nb', RandomForestClassifier())]))
                 ]
    
    """ Using for loop on piplelines, training and evaluating all the selected models.
        """
    
    for label, pipeline in pipelines:
        pipeline.fit(
                     X_train,
                     y_train
                     )
        pred = pipeline.predict(
                                X_test
                                )
        """Evaluation"""
        model_evaluation(
                         label,
                         y_test,
                         pred
                         )

if __name__ == '__main__':
    """ In case, we want to know the compute time.
        """
    start = time.time()
    """ Implementing multiprocessing
        """
    p = Process(target = main)
    p.start() 
    p.join()
    """ End time for the program and then printing final compute time.
        """
    end = time.time()
    print(end - start)
