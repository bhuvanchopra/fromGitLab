from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import os
from collections import defaultdict
from surprise import SVD, SVDpp,  NMF, KNNBasic, BaselineOnly
from surprise import NormalPredictor, KNNWithMeans, SlopeOne, CoClustering
import random
import numpy as np
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+



def benchmark(trainset, testset):
    ''' 
    Calculate the accuracy for various algorithem and print out the rmse for each algorithms
    '''
    methods= [SVD(), SVDpp(),  NMF(), KNNBasic(), BaselineOnly(), NormalPredictor(), KNNWithMeans(), SlopeOne(), CoClustering()]
    for method in methods:
        print()
        print(method)
    
        # We'll use the famous ... algorithm.
        algo = method

        # Train the algorithm on the trainset, and predict ratings for the testset
        algo.fit(trainset)

        # Than predict ratings for all pairs (u, i) that are NOT in the training set.
        predictions = algo.test(testset)

        # Then compute RMSE
        accuracy.rmse(predictions)
# 
#         #This part can be used to print the top 10 recommend product for each user in the test data set
#         top_n = get_top_n(predictions, n=10)
#         # 
#         # # Print the recommended items for each user in the test data set
#         for uid, user_ratings in top_n.items():
#             print(uid, [iid for (iid, _) in user_ratings])
       

def bestAccuracy(method, data, trainset, testset):
    ''' 
    Select your best algo with grid search.
    
    Exhaustive search over specified parameter values for an estimator.
    The parameters of the estimator are optimized by 
    cross-validated grid-search over a parameter grid.
    
    Then the accuracy of the best parameters are calculated
    
     n_epochs : The number of iteration of the SGD procedure. Default is 20.
     lr_all   : The learning rate for all parameters. Default is 0.007.
     reg_all  : The regularization term for all parameters. Default is 0.02.
     
     methods= [SVD, SVDpp] #,  NMF, KNNBasic, BaselineOnly, NormalPredictor, KNNWithMeans, SlopeOne, CoClustering]
    '''

    param_grid = {'n_epochs': [10, 15, 20], 
                  'lr_all': [0.003, 0.007, 0.009],
                  'reg_all': [0.02, 0.04, 0.06]}
#     param_grid = {'n_epochs': [25], 
#                   'lr_all': [0.009],
#                   'reg_all': [0.2]}
    print()
    print(method)
    grid_search = GridSearchCV(method, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)
    # combination of parameters that gave the best RMSE score
    print(grid_search.best_params['rmse'])
    # We can now use the algorithm that yields the best rmse:
    algo = grid_search.best_estimator['rmse']
    # retrain on the whole set training set
    algo.fit(data.build_full_trainset())    
   # Compute unbiased accuracy on test dataset.
    predictions = algo.test(testset)
    # Then compute RMSE
    validationRmse= accuracy.rmse(predictions)
    


def crossValidation(method, trainset, testset):
    from surprise.model_selection import KFold
    # define a cross-validation iterator
    kf = KFold(n_splits=3)
    algo = method
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)

        
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
    
def main(input):
    # to have reproducible experiments
    my_seed = 0
    random.seed(my_seed)
    np.random.seed(my_seed)

    
    # I convert the json file to CSV files with only 4 feature for now
    file_path = os.path.expanduser(input)

    # As we're loading a custom dataset, we need to define a reader. The
    # input file's line has the following format:
    # 'user item rating timestamp', separated by ',' characters.

    #custom dataset
    reader = Reader(line_format='item user rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)

    #splitting the data to training and test
    # sample random trainset and testset
    # test set is made of 25% of the ratings.
    trainset, testset = train_test_split(data, test_size=.25)

    #benchmark(trainset, testset)
    #crossValidation(SVD(), trainset, testset)
    
    bestAccuracy(SVD, data, trainset, testset)
    
    
    
if __name__ == '__main__':
    input = sys.argv[1]
    main(input)
    