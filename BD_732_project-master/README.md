# BD_732_project

Link for full dataset-------> https://drive.google.com/open?id=163UEgk9CPHicU7K0NNr3piUskx6NryAG

Only 1,00,000 records.
data.json link----->https://drive.google.com/open?id=1CQVcTPnSHUI6JJptLoNQ-E5R4fxjwv1M

November 23, 2018.
JQ --  DataMining-Python Jupyter Notebook
Read the json data
Created new column with rataing and delated column overall
Data cleaning with all the null values and droped nu-wanted columns - price (31.2% missing data) and brand (49.03% missing data)
dropped all the rows with any missing value
wrote a custom fucntion to parse 'helfup' column and created a new colum 'help' with 0 for No and 1 for Yes 
wrote a custom function to create a new columm for the leanght of the review
Saved the data as .csv file "data_clean.csv"
Total records: 98545
Columns: ['asin', 'categories', 'reviewText', 'reviewTime', 'reviewerID', 'reviewerName', 'summary', 'title', 'unixReviewTime', 'rating', 'help', 'rev_len']
No more missing data!

November 24, 2018:
Added new columns for aisn_code and reviewerID_code, both were needed for recomender system usinf spark 
Added jupyter notebook on EDA. Did some plots which makes sense. Feature extraction for rev_length worked! Please see the comments in the notebook for help. Those comments will also help Bhuvan while creating the Report. 



NOTES:
@Bhuvan - We may need to write a code to read data from cassendrs to do data mining. Might be code to create json? The one we are using as sample data for EDA!


November 25, 2018:
Added a spark code for Collaborative Filtering. The code requiers a file with 3 columns with integer values for user, item, rating. 
It performs the alternating least squares (ALS) and calculate the RMSE (Root mean Squared Error) for range of parameters and finds the best 
model.

November 26, 2018:
collaborative filter was implemented using surprise library. (http://surpriselib.com).
Surprise is a Python scikit building and analyzing recommender systems. 
Here are the accuracy measurment of our small dataset using various algorithem:
SVD	1.1950
SVD++	1.1931
NormalPredictor	1.6020
BaselineOnly	1.1996
KNNBasic	1.2551
KNNWithMeans	1.2562
KNNBaseline	1.1995
NMF	1.2733

SVD and SVP++ are based on the matrix factorization-based method and produce better accuracy for our recommender system.
Here is accuracy result implementing SVD where the parameters of the estimator are optimized by cross-validated grid-search over a parameter grid.

best parameters: {'n_epochs': 20, 'lr_all': 0.009, 'reg_all': 0.06}
RMSE: 0.5907

NOTES: 
@JQ: I cannot dowload the data_clean.csv. 

November 29, 2018.
There was NaN in the review column whcih was giving ====> TypeError: 'float' object is not iterable
Took lots of time to explore this error! First printed tons of messages to explore where the error is. Located the message and found it is NaN-----!

Created a pipeline to test MultinomialNB_model, LogisticRegression_model and RandomForest_model:
Resultes are below:

MultinomialNB_model
[[25404    11]
[ 7025    66]]
                    precision       recall      f1-score    support

Excellent       0.78            1.00        0.88            25415
good              0.86            0.01         0.02         7091

micro avg       0.78      0.78      0.78     32506
macro avg       0.82      0.50      0.45     32506
weighted avg       0.80      0.78      0.69     32506

LogisticRegression_model
[[24545   870]
[ 3358  3733]]
                    precision    recall  f1-score   support

Excellent       0.88      0.97      0.92     25415
good               0.81      0.53      0.64      7091

micro avg       0.87      0.87      0.87     32506
macro avg       0.85      0.75      0.78     32506
weighted avg       0.86      0.87      0.86     32506

RandomForest_model
[[25083   332]
[ 5771  1320]]
precision    recall  f1-score   support

Excellent       0.81      0.99      0.89     25415
good       0.80      0.19      0.30      7091

micro avg       0.81      0.81      0.81     32506
macro avg       0.81      0.59      0.60     32506
weighted avg       0.81      0.81      0.76     32506

4049.3036601543427  # this is time to run the code on the data that I got in jason from Bhuvan

30 Nov -- Now we have 3 tables in Cassandra. 
Keyspace is bchopra
table names are amazon_sports, sports_reviews, and sports_metadata
Anybody can see the data by connecting to cluster.

Command to run read_sports.py:
spark-submit --packages datastax:spark-cassandra-connector:2.3.1-s_2.11 --driver-memory 10g --conf "spark.driver.maxResultSize=2g" read_sports.py bchopra amazon_sports

But I couldn't print titles.
