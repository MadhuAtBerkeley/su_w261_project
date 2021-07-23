# Databricks notebook source
# MAGIC %md # Logistic Regression Algorithm
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Team 1`__

# COMMAND ----------

# MAGIC %md #Introduction
# MAGIC 
# MAGIC Logistic regression is a discriminative approach to classification problems and is admirably suited for discovering the link between features and some particular outcome. Logistic regression can be used to classify an observation into one of two classes (like ‘positive sentiment’ and ‘negative sentiment’), or into one of many classes.
# MAGIC 
# MAGIC We will leverage this algorithm as our baseline algorithm for determining flight delays, but here, we will dive more into the inner workinngs of the algorithm and write it for scalability and parallelization. The notebook is broken down into the following sections -
# MAGIC 
# MAGIC   * ... __optimization theory__ the loss function and gradient descent for logistic regression.
# MAGIC   * ... __setup__ data setup.
# MAGIC   * ... __implement__ running the basic algorithm.
# MAGIC   * ... __compare/contrast__ how L1 and L2 regularization impact model parameters & performance.

# COMMAND ----------

# MAGIC %md ## Objective
# MAGIC 
# MAGIC A machine learning system for classification has four components:
# MAGIC 1. A feature representation of the input. For each input observation \\( x(i) \\)
# MAGIC , this will be a vector of features [x1, x2,..., xn].
# MAGIC 2. A classification function that computes \\( \widehat{y} \\), the estimated class, via \\( P(y|x) \\) - essentially, the sigmoid and softmax tools for classification.
# MAGIC 3. An objective function for learning, usually involving minimizing error on training examples, as in the cross-entropy loss function.
# MAGIC 4. And finally, the algorithm for optimizing the objective function or stochastic gradient descent algorithm.
# MAGIC 
# MAGIC 
# MAGIC Logistic regression has two phases:
# MAGIC 1. Training:
# MAGIC    - We train the system ( specifically the weights \\( w_i \\) and b ) using stochastic gradient descent and the cross-entropy loss.
# MAGIC 2. Test
# MAGIC    - Given a test example x we compute \\( P(y|x) \\) and return the higher probability label \\( y = 1 \\) or \\(y = 0 \\).

# COMMAND ----------

# MAGIC %md #Optimization Theory

# COMMAND ----------

# MAGIC %md ## Logistic Loss Function
# MAGIC 
# MAGIC Given that we are looking to predict the probability of an outcome, we must model \\( p(X) \\) using a function that gives outputs between 0 and 1 for all values of \\( X \\). In logistic regression, we can express this via the conditional maximum likelihood estimation, choosing \\( w,b \\) that maximize the log probability of the true \\( y \\) labels in the training data given the observations \\( x \\).
# MAGIC 
# MAGIC 
# MAGIC Since we are looking at 2 discrete outcomes (1 or 0), this is a Bernoulli distribution, and thus express the probability of \\( p(y|x) \\) as:
# MAGIC 
# MAGIC \\( p(y|x) = \widehat y^y(1-\widehat y)^{1-y} \\)
# MAGIC 
# MAGIC Taking \\( log \\) and minimizing this log likelihood, 
# MAGIC 
# MAGIC \\( \log p(y|x) = \log [ \widehat y^y(1-\widehat y)^{1-y} ] \\) 
# MAGIC 
# MAGIC or 
# MAGIC 
# MAGIC \\( \log p(y|x) = y \log \widehat y + (1-y)log(1-\widehat y) \\) 
# MAGIC 
# MAGIC 
# MAGIC To turn this into a loss function, we will make this a minimization function (flipping the sign), and note it as the cross-entropy loss \\( L_{CE}\\) .
# MAGIC 
# MAGIC 
# MAGIC \\( L_{CE} =  - \log p(y|x) = - [ y \log \widehat y + (1-y)log(1-\widehat y) ]\\) 
# MAGIC 
# MAGIC where \\( \widehat y = \sigma(w.x + b) \\)

# COMMAND ----------

# MAGIC %md ## Gradient Descent
# MAGIC 
# MAGIC Now, that we have our loss function, we need to find a minimum of this loss function. That would allow us to find optimal weights for a logistic regression model. We will represent weights as \\( \theta \\).
# MAGIC 
# MAGIC 
# MAGIC \\( \widehat \theta = argmin \frac{1}{m}\sum_{i=1}^{m} L(f(x_i, \theta), y_i) \\)
# MAGIC 
# MAGIC The loss function is conveniently convex and thus has just one minimum. From an algorithm perspective, we are looking to move in the direction of fastest descent, as in \\( \frac{d}{dw} f(x;w) \\) weighted by a learning rate \\( \eta \\). Given that there will be multiple such feature variables \\( x \\), we would ultimately have a dimension vector that has the slope for each \\( x \\) as a partial derivative of the loss function. To summarize that, we will represent gradient descent \\( L(f(x, \theta), y) \\) in an equation for updating \\( \theta \\) as:
# MAGIC \\(   \\)
# MAGIC 
# MAGIC \\( \theta_{t+1} = \theta_t - \nabla L(f(x, \theta), y) \\)
# MAGIC 
# MAGIC 
# MAGIC For updating \\( \theta \\), we need a definition for our gradient function, which we can obtain by taking the derivative of our loss function from earlier.
# MAGIC 
# MAGIC \\( L_{CE} =  - [ y \log \sigma(w.x + b) + (1-y)log(1-\sigma(w.x + b)) ]\\) 
# MAGIC 
# MAGIC \\( \frac{ \partial L_{CE}(\widehat{y},t) }{\partial w_j} = [\sigma(w.x + b) - y]x_j \\)
# MAGIC 
# MAGIC We will use this form in calculating the optimal weights during the execution of the logistic regression algorithm.

# COMMAND ----------

# MAGIC %md #Notebook Set-Up
# MAGIC 
# MAGIC We will leverage the data about red and white Portuguese wines (http://archive.ics.uci.edu/ml/datasets/Wine+Quality) made available by UC Irvine as part of a public repository of Machine Learning datasets.

# COMMAND ----------

# imports
import re
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os

# COMMAND ----------

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, expr, unix_timestamp, avg, round, lag
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.ml.feature import OneHotEncoder , StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_colwidth', 999)
pd.set_option('display.max_columns', 999)

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, expr, unix_timestamp, avg, round, lag, to_date, row_number, explode, array, lit
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType

import numpy as np
import math

#import mlflow
#import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
 
from numpy import savetxt
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# COMMAND ----------

df = sqlContext.sql("""SELECT * FROM airlines_with_features_sql""")

# COMMAND ----------

print(df.columns)

# COMMAND ----------

sc = spark.sparkContext
df = sqlContext.sql("""SELECT * FROM airlines_with_features_sql""")
df = df.where((col('DEP_DELAY').isNotNull()) & (col('ORIGIN') == 'ORD') | (col('ORIGIN') == 'ATL'))


features = ['MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_CARRIER', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'CRS_ARR_TIME', 'ORIGIN', 'DEST_AIRPORT_ID',
           'Src_WND_0', 'Src_WND_3', #'Src_WND_3', 'Src_WND_4', 
            'Src_TMP_0', 'Src_VIS_0', #'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1',  'Src_VIS_3',  'Src_TMP_1',  
           'Src_DEW_0', #'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 
            #'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 
            'DEP_DEL15']

df = df.select(*features).na.drop("any")


# COMMAND ----------

from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import udf, struct
from pyspark.sql.functions import weekofyear

def quantize_time(x):
    xH = int(x/100)
    xM = x%100
    xM = 15*int((xM+7)/15)
    return int(xH*100+xM)
  
def flight_duration(x, y):
   
    xH = int(x/100)
    yH = int(y/100)
    xM = x % 100
    yM = y % 100
    if(x > y):
        fl_time = int((xH-yH)*60+(xM-yM))
    else:
        xM = 60 + xM
        fl_time = int((23+xH-yH)*60+(xM-yM))
      
    return fl_time
  
def origin_map(x):
    return (1 if (x=='ATL') else 2)
   

comp_fn = udf(flight_duration, IntegerType())
quant_time = udf(quantize_time, IntegerType())
origin_fn = udf(origin_map, IntegerType())


df = df.withColumn("FLIGHT_TIME_MINS", comp_fn("CRS_ARR_TIME","CRS_DEP_TIME")) \
       .withColumn("WEEK_OF_YEAR", weekofyear("FL_DATE")) \
       .withColumn("CRS_DEP_TIME_QUANT", quant_time("CRS_DEP_TIME")) \
       .withColumn("ORIGIN", origin_fn("ORIGIN"))

print(df.columns)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
df_columns= ['OP_CARRIER', 'CRS_DEP_TIME_QUANT', 'ORIGIN', 'DEST_AIRPORT_ID', 'DAY_OF_WEEK', 'WEEK_OF_YEAR', 'FLIGHT_TIME_MINS', 'DEP_DEL15']

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in df_columns ]


pipeline = Pipeline(stages=indexers)
df= pipeline.fit(df).transform(df)

# COMMAND ----------

print(df.columns)

# COMMAND ----------

df_delay_day = df.groupBy("DAY_OF_WEEK").sum("DEP_DEL15")
df_delay_week = df.groupBy("WEEK_OF_YEAR").sum("DEP_DEL15")
df_delay_carr = df.groupBy("OP_CARRIER").sum("DEP_DEL15")
df_delay_time = df.groupBy("CRS_DEP_TIME_QUANT").sum("DEP_DEL15")
df_delay_dst = df.groupBy("DEST_AIRPORT_ID").sum("DEP_DEL15")

avg_delay_day = df_delay_day.select(['DAY_OF_WEEK', 'sum(DEP_DEL15)']).rdd.map(lambda x: (x[0], x[1])).collect()
avg_delay_week = df_delay_week.select(['WEEK_OF_YEAR', 'sum(DEP_DEL15)']).rdd.map(lambda x: (x[0], x[1])).collect()
avg_delay_carr = df_delay_carr.select(['OP_CARRIER', 'sum(DEP_DEL15)']).rdd.map(lambda x:(x[0], x[1])).collect()
avg_delay_time = df_delay_time.select(['CRS_DEP_TIME_QUANT', 'sum(DEP_DEL15)']).rdd.map(lambda x:(x[0], x[1])).collect()
avg_delay_dst = df_delay_dst.select(['DEST_AIRPORT_ID', 'sum(DEP_DEL15)']).rdd.map(lambda x:(x[0], x[1])).collect()
                                                                                  
avg_delay_day = {x[0]:x[1] for x in avg_delay_day}    
avg_delay_week = {x[0]:x[1] for x in avg_delay_week}                                                                                   
avg_delay_carr = {x[0]:x[1] for x in avg_delay_carr} 
avg_delay_time = {x[0]:x[1] for x in avg_delay_time} 
avg_delay_dst = {x[0]:x[1] for x in avg_delay_dst}                                                                                   
#avg_delay_dst = df_delay_carr.select([]).rdd.map(lambda x:{}).collect()
#avg_delay_carr = df_delay_carr.select([]).rdd.map(lambda x:{}).collect()
#print(avg_delay_week)
#print(avg_delay_carr)
#print(avg_delay_time)
#print(avg_delay_dst)

# COMMAND ----------

features = ['OP_CARRIER', 'CRS_DEP_TIME_QUANT', 'ORIGIN', 'DEST_AIRPORT_ID', 'DAY_OF_WEEK', 'WEEK_OF_YEAR', 'FLIGHT_TIME_MINS',
            'OP_CARRIER_index', 'CRS_DEP_TIME_QUANT_index', 'ORIGIN_index', 'DEST_AIRPORT_ID_index', 'DAY_OF_WEEK_index', 'WEEK_OF_YEAR_index', 'FLIGHT_TIME_MINS_index',
            'Src_WND_0', 'Src_WND_3', #'Src_WND_3', 'Src_WND_4', 
            'Src_TMP_0', 'Src_VIS_0', #'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1',  'Src_VIS_3',  'Src_TMP_1',  
            'Src_DEW_0', #'Src_DEW_1', #'Src_SLP_0', 'Src_SLP_1', 
            'DEP_DEL15_index']


def target_en_day(x):
   return avg_delay_day[x]
  
def target_en_week(x):
   return avg_delay_week[x]
  
def target_en_time(x):
   return avg_delay_time[x]
  
def target_en_dst(x):
   return avg_delay_dst[x]
  
def target_en_carr(x):
   return avg_delay_carr[x]  
  
call_fn_day = udf(target_en_day, DoubleType())  
call_fn_week = udf(target_en_week, DoubleType())
call_fn_time = udf(target_en_time, DoubleType())
call_fn_dst = udf(target_en_dst, DoubleType())
call_fn_carr = udf(target_en_carr, DoubleType())


df1 = df.select(*features) \
        .withColumn('OP_CARRIER', call_fn_carr('OP_CARRIER')) \
        .withColumn('CRS_DEP_TIME_QUANT', call_fn_time('CRS_DEP_TIME_QUANT')) \
        .withColumn('WEEK_OF_YEAR', call_fn_week('WEEK_OF_YEAR')) \
        .withColumn('DAY_OF_WEEK', call_fn_day('DAY_OF_WEEK')) \
        .withColumn('DEST_AIRPORT_ID', call_fn_dst('DEST_AIRPORT_ID')) 

# COMMAND ----------

def sample_df(df,ratio,sample_flag):
  
  df_minority = df.where(col('DEP_DEL15') == 1)
  df_majority = df.where(col('DEP_DEL15') == 0)
  
  # Oversample
  if sample_flag == 1:
    # range based on calculated ratio
    y = range(ratio)

    # duplicate the minority class rows
    df_duplicate = df_minority.withColumn('temp', explode(array([lit(x) for x in y]))).drop('temp')

    # combine oversampled delayed flights with not delayed flight records
    df_oversampled = df_majority.union(df_duplicate)

    # check the results

    not_delayed = df_oversampled.where(col('DEP_DEL15') == 0).count()
    delayed = df_oversampled.where(col('DEP_DEL15') == 1).count()

    print('Oversampling minority class results in:')
    print(f'Number of flights delayed: {delayed}')
    print(f'Number of flights not-delayed: {not_delayed}')
    print(f'Ratio: {not_delayed / delayed}')
    
    return df_oversampled
  
  # Undersample
  if sample_flag == 2:
    # undersample the records corresponding to not delayed flights according to the ratio 1:4
    df_sampled_major = df_majority.sample(False, 1/ratio)

    # create new dataframe with undersampled DEP_DEL15=0 and all records DEP_DEL15=1
    df_undersampled = df_sampled_major.union(df_minority)

    # check the results

    not_delayed = df_undersampled.where(col('DEP_DEL15') == 0).count()
    delayed = df_undersampled.where(col('DEP_DEL15') == 1).count()

    print('Undersampling majority class results in:')
    print(f'Number of flights delayed: {delayed}')
    print(f'Number of flights not-delayed: {not_delayed}')
    print(f'Ratio: {not_delayed / delayed}')
    
    return df_undersampled
  
  return df

# COMMAND ----------

features = ['OP_CARRIER', 'CRS_DEP_TIME_QUANT', 'ORIGIN', 'DEST_AIRPORT_ID', 'DAY_OF_WEEK', 'WEEK_OF_YEAR', 'FLIGHT_TIME_MINS', 
            'Src_WND_0', 'Src_WND_3', #'Src_WND_3', 'Src_WND_4', 
            'Src_TMP_0', 'Src_VIS_0', #'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1',  'Src_VIS_3',  'Src_TMP_1',  
            'Src_DEW_0', #'Src_DEW_1', #'Src_SLP_0', 'Src_SLP_1', 
            'DEP_DEL15']


# Generate 80/20 (pseudo)random train/test split - RUN THIS CELL AS IS
df1 = df1.select(*features)
trainRDD, heldOutRDD = df1.randomSplit([0.8,0.2], seed = 1)
print(f"... held out {heldOutRDD.count()} records for evaluation and assigned {trainRDD.count()} for training.")

ratio = 4
sample_flag = 2
trainRDD = sample_df(trainRDD,ratio,sample_flag)

trainRDD = trainRDD.rdd.map(lambda x: list(x))
trainRDDCached = trainRDD.map(lambda x: (np.array(x[0:-1]), x[-1])).cache()

heldOutRDD = heldOutRDD.rdd.map(lambda x: list(x))
heldOutRDDCached = heldOutRDD.map(lambda x: (np.array(x[0:-1]), x[-1])).cache()

trainRDDCached.take(1)
                 

# COMMAND ----------

# Setting up notebook to leverage user dir
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
algo_path = userhome + "/final_project_algo/" 
algo_path_open = '/dbfs' + algo_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(algo_path)

total = 0
DATA_PATH = 'dbfs:/mnt/mids-w261/'
for item in dbutils.fs.ls(DATA_PATH):
  total = total+item.size
total

sc = spark.sparkContext
#spark
#display(dbutils.fs.ls('file:/tmp'))  

# COMMAND ----------

# For the toy dataset, we will make use of the red & white wine files

!wget -P /tmp https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!wget -P /tmp https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

dbutils.fs.cp('file:/tmp/winequality-red.csv', algo_path, recurse=True)  
dbutils.fs.cp('file:/tmp/winequality-white.csv', algo_path, recurse=True)  
display(dbutils.fs.ls('dbfs:/user/amitk@berkeley.edu/final_project_algo'))

# COMMAND ----------

# MAGIC %md
# MAGIC We will combine the white and red wines datasets and classify the wines as either good quality or otherwise based on the quality index score ( with 1 for quality index > 5 for a "good" classification ). We will obvioulsy remove this index from the dataset to remove this direct correlation.

# COMMAND ----------

# extract header fields - RUN THIS CELL AS IS
header = !head -n 1 '/tmp/winequality-red.csv'
header = header[0]
FIELDS = ['color'] + re.sub('"', '', header).split(';')

# load the raw data into an RDD - RUN THIS CELL AS IS
redsRDD = sc.textFile('dbfs:/user/amitk@berkeley.edu/final_project_algo/winequality-red.csv')\
            .filter(lambda x: x != header)\
            .map(lambda x: '1;' + x) # set first field 1 to indicate red wine
whitesRDD = sc.textFile('dbfs:/user/amitk@berkeley.edu/final_project_algo/winequality-white.csv')\
              .filter(lambda x: x != header)\
              .map(lambda x: '0;' + x) # set first field 0 to indicate white wine


# Generate 80/20 (pseudo)random train/test split - RUN THIS CELL AS IS
trainRDD, heldOutRDD = redsRDD.union(whitesRDD).randomSplit([0.8,0.2], seed = 1)
print(f"... held out {heldOutRDD.count()} records for evaluation and assigned {trainRDD.count()} for training.")

# helper function that also adds in the classification of high/low quality wines
# based on quality
def parse(line):
    """
    Map record_csv_string --> (tuple,of,fields)
    """    
    fields = np.array(line.split(';'), dtype = 'float')
    features,quality = fields[:-1], (1 if fields[-1] >=6 else 0)
    return(features, quality)
  
# cache the training set - RUN THIS CELL AS IS 
trainRDDCached = trainRDD.map(parse).cache()  
heldOutRDDCached = heldOutRDD.map(parse).cache()
#trainRDDCached.take(1)

# COMMAND ----------

# part a - take a 1000 point sample for EDA (RUN THIS CELL AS IS)
sample = np.array(trainRDDCached.map(lambda x: np.append(x[0], [x[1]]))
                                .takeSample(False, 1000))
sample_df = pd.DataFrame(np.array(sample), columns = FIELDS)

# COMMAND ----------

# MAGIC %md ## Basic EDA

# COMMAND ----------

# part a - take a look at histograms for each feature (RUN THIS CELL AS IS)
sample_df[FIELDS[:-1]].hist(figsize=(15,15), bins=15)
display(plt.show())

# COMMAND ----------

# MAGIC %md # Running the algorithm

# COMMAND ----------

# part a - mean and variance of the outcome variable 

meanQuality = trainRDDCached.map(lambda x: x[1]).mean()
varQuality = trainRDDCached.map(lambda x: x[1]).variance()

print(f"Mean: {meanQuality}")
print(f"Variance: {varQuality}")

# COMMAND ----------

# MAGIC %md ## Cross-Entropy Loss

# COMMAND ----------

def sigmoid(x):
  '''
  Compute the sigmoid for x - replicating 1, 0 state classification
  '''
  return 1/(1 + np.exp(-x))

def predicted(W,x):
  logit = W.dot(x)
  return sigmoid(logit)

def crossEntropyLoss(dataRDD, W):
    """
    Compute mean squared error.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    #augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    # take the dot product for the likelihood function
    residuals = dataRDD.map(lambda x: -x[1]*np.log(predicted(W,x[0]))\
                                  - (1-x[1])*np.log(1 -predicted(W,x[0]))).collect()
    
    # take the mean of the residuals for the loss
    loss =  np.mean(residuals)
    return loss
  
  
  

# COMMAND ----------

# MAGIC %md ## Performance Metric

# COMMAND ----------

def predictLabel(testRDD, W, threshold):
    predRDD = testRDD.map(lambda x: 1 if (predicted(W,x[0]) > threshold) else 0)
    return predRDD
   
  
def computeMetric(testRDD, W, threshold):
    predRDD = predictLabel(testRDD, W, threshold)   
    trueRDD = testRDD.map(lambda x: x[1])
    return 

# COMMAND ----------

# part e - define your baseline model here
BASELINE = np.append([meanQuality], np.zeros(len(trainRDDCached.take(1)[0][0])))
#augmentedRDD = trainRDDCached.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()

# COMMAND ----------

# MAGIC %md ## Gradient Descent

# COMMAND ----------

# part b - function to perform a single GD step
def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one LR gradient descent step/update.
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        new_model - (array) updated coefficients, bias at index 0
    """
    # add a bias 'feature' of 1 at index 0
    #augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()    
    #n = augmentedData.count()
    inv_N = grad_scale.value
    gradient = dataRDD.map(lambda x: np.dot((predicted(W,x[0]) - x[1]),x[0]) )\
                            .reduce(lambda x,y: x+y)

    # new_coefficients = old_coefficients - learningrate * gradient
    new_model = W - np.multiply(learningRate, inv_N * gradient)   
    return new_model

# COMMAND ----------

# MAGIC %md ### Normalize Data and Run 

# COMMAND ----------

# Helper function to normalize the data
def normalize(dataRDD):
    """
    Scale and center data round mean of each feature.
    Args:
        dataRDD - records are tuples of (features_array, y)
    Returns:
        normedRDD - records are tuples of (features_array, y)
    """
    featureMeans = dataRDD.map(lambda x: x[0]).mean()
    featureStdev = np.sqrt(dataRDD.map(lambda x: x[0]).variance())    
    normedRDD = dataRDD.map(lambda x: ((x[0] - featureMeans)/featureStdev,x[1]))
    
    return normedRDD

# COMMAND ----------

# part d - cache normalized data (RUN THIS CELL AS IS)
normedRDD = normalize(trainRDDCached).cache()

# COMMAND ----------

# MAGIC %%time
# MAGIC # part e - take a look at a few GD steps w/ normalized data  (RUN THIS CELL AS IS)
# MAGIC nSteps = 50
# MAGIC model = BASELINE
# MAGIC augmentedRDD = normedRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()
# MAGIC grad_scale = sc.broadcast(1.0/trainRDDCached.count())
# MAGIC print(f"BASELINE:  Loss = {crossEntropyLoss(augmentedRDD,model)}")
# MAGIC for idx in range(nSteps):
# MAGIC     print("----------")
# MAGIC     print(f"STEP: {idx+1}")
# MAGIC     model = GDUpdate(augmentedRDD, model, 0.1)
# MAGIC     loss = crossEntropyLoss(augmentedRDD, model)
# MAGIC     print(f"Loss: {loss}")
# MAGIC     #print(f"Model: {[round(w,3) for w in model]}")

# COMMAND ----------

W = model
threshold = 0.515
def normalizeTestSet(trainRDD, testRDD):
    featureMeans = trainRDD.map(lambda x: x[0]).mean()
    featureStdev = np.sqrt(trainRDD.map(lambda x: x[0]).variance())    
    normedRDD = testRDD.map(lambda x: ((x[0] - featureMeans)/featureStdev,x[1]))
    
    return normedRDD
    
#heldOutRDDCached = heldOutRDD.map(parse).cache()    
normedTestRDD = normalizeTestSet(trainRDDCached, heldOutRDDCached).cache()
augHeldOutRDD = normedTestRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
predRDD = augHeldOutRDD.map(lambda x: 1 if (predicted(W,x[0]) > threshold) else 0)

trueRDD = augHeldOutRDD.map(lambda x: x[1])


# COMMAND ----------


import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# COMMAND ----------

from sklearn.metrics import f1_score

y_pred = predRDD.collect()
y_true = trueRDD.collect()

print(f1_score(y_true, y_pred, average='micro'))
cnf_matrix = confusion_matrix(y_true, y_pred)
plt.figure()
class_names = ['no_delay', 'delay']
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, without normalization')
plt.show()

# COMMAND ----------

from sklearn.metrics import f1_score
print(f1_score(true_l, pred, average='micro'))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Assessing the performance of the model.
# MAGIC 
# MAGIC Printing out the loss as we perform each gradient descent step allows us to confirm that our Gradient Descent code appears to be working, but this number doesn't accurately reflect "how good" our model is. We will plot error curves for a test and training set in order to gauge model performance. Note that although we split out a test & train set when we first loaded the data... in the spirit of keeping that 20% truly 'held out', we'll make an additional split by dividing the existing training set into two smaller RDDs.

# COMMAND ----------

def GradientDescent(trainRDD, testRDD, wInit, nSteps = 20, 
                    learningRate = 0.1, verbose = False):
    """
    Perform nSteps iterations of OLS gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
  
    for idx in range(nSteps): 
        
        ############## YOUR CODE HERE #############
        model = GDUpdate(trainRDD, model, learningRate)
        training_loss = crossEntropyLoss(trainRDD, model) 
        test_loss = crossEntropyLoss(testRDD, model) 

        ############## (END) YOUR CODE #############
        
        # keep track of test/train loss for plotting
        train_history.append(training_loss)
        test_history.append(test_loss)
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"training loss: {training_loss}")
            print(f"test loss: {test_loss}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, model_history

# COMMAND ----------

# plot error curves - RUN THIS CELL AS IS
def plotErrorCurves(trainLoss, testLoss, title = None):
    """
    Helper function for plotting.
    Args: trainLoss (list of MSE) , testLoss (list of MSE)
    """
    fig, ax = plt.subplots(1,1,figsize = (16,8))
    x = list(range(len(trainLoss)))[1:]
    ax.plot(x, trainLoss[1:], 'k--', label='Training Loss')
    ax.plot(x, testLoss[1:], 'r--', label='Test Loss')
    ax.legend(loc='upper right', fontsize='x-large')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Squared Error')
    if title:
        plt.title(title)
    display(plt.show())

# COMMAND ----------

# MAGIC %md ## With Normalization

# COMMAND ----------

# run 50 iterations (RUN THIS CELL AS IS)
wInit = BASELINE
trainRDD, testRDD = augmentedRDD.randomSplit([0.8,0.2], seed = 4)
start = time.time()
MSEtrain, MSEtest, models = GradientDescent(trainRDD, testRDD, wInit, nSteps = 50)
print(f"\n... trained {len(models)} iterations in {time.time() - start} seconds")

# COMMAND ----------

# Error curves based on normalized data
plotErrorCurves(MSEtrain, MSEtest, title = 'Logistic Regression with normalized data' )

# COMMAND ----------

# MAGIC %md # Regularization.
# MAGIC 
# MAGIC The goal, as always, is to build a linear model that will extend well to unseen data. Chosing the right combination of features to optimize generalizability can be extremely computationally costly given that there are \\(2^{p}\\) potential models that can be built from \\(p\\) features. Traditional methods like forward selection would involve iteratively testing these options to asses which combinations of features achieve a statistically significant prediction.
# MAGIC 
# MAGIC Ridge Regression and Lasso Regression are two popular alternatives for Logistic Regression, which enable us to train generalizable models without the trouble of forward selection and/or manual feature selection.  Both methods take advantage of the bias-variance tradeoff by _shrinking_ the model coefficients towards 0 which reduces the variance of our model with little increase in bias. In practice this 'shrinkage' is achieved by adding a penalty (a.k.a. 'regularization') term to the means squared error loss function. In this question you will implement Gradient Descent with ridge and lasso regularization.

# COMMAND ----------

def GDUpdate_wReg(dataRDD, W, learningRate = 0.1, regType = None, regParam = 0.1):
    """
    Perform one gradient descent step/update with ridge or lasso regularization.
    Args:
        dataRDD - tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        learningRate - (float) defaults to 0.1
        regType - (str) 'ridge' or 'lasso', defaults to None
        regParam - (float) regularization term coefficient
    Returns:
        model   - (array) updated coefficients, bias still at index 0
    """
    # augmented data
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))    
    new_model = None
    
    # calculate the gradient
    n = augmentedData.count()
    gradient = augmentedData.map(lambda x: np.dot(predicted(W,x[0]) - x[1],x[0]) )\
                            .reduce(lambda x,y: x+y)    
    # depending upon regularization, apply different penalty
    
    if regType == 'ridge':
        l2_reg_term = np.multiply(2 * regParam , W[1:])
        l2_reg_term = np.append([0], l2_reg_term)
        gradient_term = np.add((1/n) * gradient, l2_reg_term)
        new_model = W - np.multiply(learningRate, gradient_term)
        
    elif regType == 'lasso':
        l1_reg_term = regParam * np.sign(W[1:])
        l1_reg_term = np.append([0], l1_reg_term)        
        gradient_term = np.add((1/n) * gradient, l1_reg_term)
        new_model = W - np.multiply(learningRate, gradient_term)
        
    else:
        gradient_term = (2/n) * gradient
        new_model = W - np.multiply(learningRate, gradient_term)
        
    return new_model

# COMMAND ----------

# ridge/lasso gradient descent function
def GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 20, learningRate = 0.1,
                         regType = None, regParam = 0.1, verbose = False):
    """
    Perform nSteps iterations of regularized gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps):  
        # update the model
        model = GDUpdate_wReg(trainRDD, model, learningRate, regType, regParam)
        
        # keep track of test/train loss for plotting
        train_history.append(crossEntropyLoss(trainRDD, model))
        test_history.append(crossEntropyLoss(testRDD, model))
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"training loss: {training_loss}")
            print(f"test loss: {test_loss}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, model_history

# COMMAND ----------

# run 50 iterations of ridge
wInit = BASELINE
trainRDD, testRDD = augmentedRDD.randomSplit([0.8,0.2], seed = 5)
start = time.time()
ridge_results = GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 50, 
                                     regType='ridge', regParam = 0.05 )
print(f"\n... trained {len(ridge_results[2])} iterations in {time.time() - start} seconds")

# COMMAND ----------

# part d - save and display ridge results
trainLoss, testLoss, models = ridge_results
plotErrorCurves(trainLoss, testLoss, title = 'Ridge Regression Error Curves' )

# COMMAND ----------

# run 50 iterations of lasso
wInit = BASELINE
trainRDD, testRDD = augmentedRDD.randomSplit([0.8,0.2], seed = 5)
start = time.time()
lasso_results = GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 50,
                                     regType='lasso', regParam = 0.05)
print(f"\n... trained {len(lasso_results[2])} iterations in {time.time() - start} seconds")

# COMMAND ----------

# display lasso results
trainLoss, testLoss, models = lasso_results
plotErrorCurves(trainLoss, testLoss, title = 'Lasso Regression Error Curves' )

# COMMAND ----------

last_model = model
print(model)
