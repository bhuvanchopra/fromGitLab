import sys
from pyspark.sql import SparkSession, functions, types
from cassandra.cluster import Cluster, BatchStatement, ConsistencyLevel

# Working with the production/reliable Cassandra cluster
cluster_seeds = ['199.60.17.188', '199.60.17.216']
spark = SparkSession.builder.appName('Spark Cassandra') \
.config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')


def main(inputs1, inputs2, keyspace, table1, table2):

    data_schema = types.StructType([
    types.StructField('reviewerID', types.StringType(), True),
    types.StructField('asin', types.StringType(), True),
    types.StructField('reviewerName', types.StringType(), True),
    types.StructField('helpful', types.StringType(), True),
    types.StructField('reviewText', types.StringType(), True),
    types.StructField('overall', types.DoubleType(), True),
    types.StructField('summary', types.StringType(), True),
    types.StructField('unixReviewTime', types.TimestampType(), True),
    types.StructField('reviewTime', types.StringType(), True)])

    metadata_schema = types.StructType([
    types.StructField('asin', types.StringType(), True),
    types.StructField('title', types.StringType(), True),
    types.StructField('price', types.DoubleType(), True),
    types.StructField('brand', types.StringType(), True),
    types.StructField('categories', types.StringType(), True)])

    reviews_data = spark.read.json(inputs1, schema=data_schema)

    metadata = spark.read.json(inputs2, schema=metadata_schema).na.drop(subset=['asin'])

    reviews_data.write.format("org.apache.spark.sql.cassandra").mode('overwrite') \
    .option('confirm.truncate', True).options(table=table1, keyspace=keyspace).save()

    metadata.write.format("org.apache.spark.sql.cassandra").mode('overwrite') \
    .option('confirm.truncate', True).options(table=table2, keyspace=keyspace).save()


if __name__ == '__main__':
    inputs1 = sys.argv[1]
    inputs2 = sys.argv[2]
    keyspace = sys.argv[3]
    table1 = sys.argv[4]
    table2 = sys.argv[5]
    main(inputs1, inputs2, keyspace, table1, table2)
