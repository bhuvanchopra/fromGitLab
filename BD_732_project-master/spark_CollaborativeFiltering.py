import itertools
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('review_data').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
from pyspark.sql import types
spark.sparkContext.setLogLevel('WARN')
sc=spark.sparkContext
sc.setCheckpointDir('checkpoint/') 


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
def computeRmse(model, data):
    """
    Compute RMSE (Root mean Squared Error).
    """
    predictions = model.transform(data)
    rmse = evaluator.evaluate(predictions)
    return rmse     


# requiers a file with 3 columns with integer values for user, item, rating 
lines = spark.read.text("ratings.csv").rdd
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))                         
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])


# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
# setting nonnegative=False reduce the rmse


ranks = [5, 10, 15]
lambdas = [0.01, 0.02, 0.04]
numIters = [10, 20, 30]
bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1

for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):

    als = ALS(rank=rank, maxIter=numIter, regParam=lmbda, 
        numUserBlocks=10, numItemBlocks=10, alpha=1.0,seed=1,  
        userCol="userId", itemCol="movieId", ratingCol="rating",  
        nonnegative=True,  coldStartStrategy="drop", implicitPrefs=False,
        checkpointInterval=10, intermediateStorageLevel="MEMORY_AND_DISK", finalStorageLevel="MEMORY_AND_DISK")
    
    model=als.fit(training)
    validationRmse = computeRmse(model, test)
    
    
    print( "RMSE (validation) = %f for the model trained with " % validationRmse + \
            "rank = %d, lambda = %.3f, and numIter = %d." % (rank, lmbda, numIter))
    if (validationRmse < bestValidationRmse):
        bestModel = model
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lmbda
        bestNumIter = numIter
 
model=bestModel
print( "the best RMSE (validation) = %f for the model trained with " % bestValidationRmse + \
            "rank = %d, lambda = %.1f, and numIter = %d." % (bestRank, bestLambda, bestNumIter))
    
# Generate top 10 items recommendations for each user
userRecs = model.recommendForAllUsers(10).show()


# Generate top 10 user recommendations for each item
itemRecs = model.recommendForAllItems(10).show()




