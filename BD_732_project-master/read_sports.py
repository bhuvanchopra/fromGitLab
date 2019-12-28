import sys
import itertools
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession, functions, types
from cassandra.cluster import Cluster, BatchStatement, ConsistencyLevel

cluster_seeds = ['199.60.17.188', '199.60.17.216']
spark = SparkSession.builder.appName('Spark Cassandra') \
.config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')


evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction")

def computeRmse(model, data):
    """
    Compute RMSE (Root mean Squared Error).
    """
    predictions = model.transform(data)
    rmse = evaluator.evaluate(predictions)
    return rmse

def main(keyspace, table):

    df = spark.read.format("org.apache.spark.sql.cassandra") \
    .options(table=table, keyspace=keyspace).load().cache()
    asin_indexer = StringIndexer(inputCol="asin", outputCol="asin_code")
    rID_indexer = StringIndexer(inputCol="reviewerID",outputCol='reviewerID_code')
    df = asin_indexer.fit(df).transform(df)
    df = rID_indexer.fit(df).transform(df)

    df.createOrReplaceTempView("df")
    df2 = df.select(['asin_code','reviewerID_code','overall']).cache()
    df2.describe().show()
    (training, test) = df2.randomSplit([0.8, 0.2])

    als = ALS(rank=5, maxIter=10, regParam=0.01,
        numUserBlocks=10, numItemBlocks=10, alpha=1.0,seed=1,
        userCol="reviewerID_code", itemCol="asin_code", ratingCol="overall",
        nonnegative=True,  coldStartStrategy="drop", implicitPrefs=False,
        checkpointInterval=10, intermediateStorageLevel="MEMORY_AND_DISK", finalStorageLevel="MEMORY_AND_DISK")

    model=als.fit(training)
    validationRmse = computeRmse(model, test)

    print( "RMSE (validation) = %f for the model trained with " % validationRmse)

    # Generate top 10 items recommendations for each user
    userRecs = model.recommendForAllUsers(10).show()
    # Generate top 10 user recommendations for each item
    itemRecs = model.recommendForAllItems(10).show()

if __name__ == '__main__':
    keyspace = sys.argv[1]
    table = sys.argv[2]
    main(keyspace, table)
