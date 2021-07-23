# Databricks notebook source
# MAGIC %md # TEMP NOTEBOOK FOR MODELS
# MAGIC ## Links
# MAGIC - https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
# MAGIC - https://spark.apache.org/docs/1.4.1/ml-features.html#normalizer
# MAGIC - https://spark.apache.org/docs/latest/ml-pipeline.html
# MAGIC - https://spark.apache.org/docs/latest/mllib-feature-extraction.html#example-1
# MAGIC - https://spark.apache.org/docs/1.5.0/ml-features.html#vectorassembler
# MAGIC - https://www.timlrx.com/blog/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator
# MAGIC - http://spark.apache.org/docs/2.1.0/api/python/pyspark.ml.html?highlight=featureimportance#pyspark.ml.classification.RandomForestClassificationModel.featureImportances
# MAGIC - https://runawayhorse001.github.io/LearningApacheSpark/classification.html

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

# MAGIC %md ## CONFIG

# COMMAND ----------

# =======================================================
# 3 is our agreed method to split into train/test/dev
train_simple = 3

# =======================================================
# Sample Method for class inbalance 
# 0 no change
# 1 oversample
# 2 undersample
sample_flag = 0

# COMMAND ----------

# MAGIC %md ## Functions

# COMMAND ----------

#SOURCE https://runawayhorse001.github.io/LearningApacheSpark/classification.html

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

def get_dtype(df,coltype):
  col_list = []
  for name,dtype in df.dtypes:
    if dtype == coltype:
      col_list.append(name)
  return col_list

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

sc = spark.sparkContext
sqlContext = SQLContext(sc)

# COMMAND ----------

#df = spark.read.parquet("/mnt/Azure/processed_data/airlines_with_features.parquet/")
#df.createOrReplaceTempView("df")

# COMMAND ----------

#%sql DROP TABLE IF EXISTS airlines_with_features_sql; CREATE TABLE airlines_with_features_sql AS SELECT * FROM df

# COMMAND ----------

df = sqlContext.sql("""SELECT *,HOUR(dep_datetime_scheduled) H_DEP FROM airlines_with_features_sql""")

# COMMAND ----------

#df = df.sample(fraction=.01, seed=3)
print(df.count())

# COMMAND ----------

# MAGIC %md # DATAPREP

# COMMAND ----------

# MAGIC %md ### Oversample Ratio

# COMMAND ----------

# calculate the ratio of the classes
df_minority = df.where(col('DEP_DEL15') == 1)
df_majority = df.where(col('DEP_DEL15') == 0)
major = df_majority.count()
minor = df_minority.count()
ratio = int(major/minor)
print(f'There are {ratio} times more flights not delayed than flights delayed')
print(f'Number of records of flights delayed: {minor}')
print(f'Number of records of flights not delayed: {major}')

# COMMAND ----------

# MAGIC %md ### NOTE: Weather Data has nulls, we need to fill in with average after the join in the ETL.

# COMMAND ----------

df = df.na.fill(0)

# COMMAND ----------

print(get_dtype(df,'int'))
print("================================")
print(get_dtype(df,'double'))
print("================================")
print(get_dtype(df,'long'))
print("================================")
print(get_dtype(df,'string'))

# COMMAND ----------

# MAGIC %md ### FEATURES USED

# COMMAND ----------

numericCols =   ['ORPageRank', 'proxy_origin', 'proxy_dest','carrier_delay','airport_delay','H_DEP','Src_WND_0', 'Src_WND_1', 'Src_WND_3', 'Src_WND_4', 'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1', 'Src_VIS_2', 'Src_VIS_3', 'Src_TMP_0', 'Src_TMP_1', 'Src_DEW_0', 'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 'Dst_WND_0', 'Dst_WND_1', 'Dst_WND_3', 'Dst_WND_4', 'Dst_CIG_0', 'Dst_CIG_1', 'Dst_VIS_0', 'Dst_VIS_1', 'Dst_VIS_2', 'Dst_VIS_3', 'Dst_TMP_0', 'Dst_TMP_1', 'Dst_DEW_0', 'Dst_DEW_1', 'Dst_SLP_0', 'Dst_SLP_1', 'Dst_GA1_0', 'Dst_GA1_1', 'Dst_GA1_2', 'Dst_GA1_3', 'Dst_GA1_4', 'Dst_GA1_5','DEP_CNT', 'DST_CNT', 'DESTPageRank', 'DAY_OF_MONTH', 'DAY_OF_WEEK','YEAR', 'QUARTER', 'MONTH' , 'LATE_AIRCRAFT_DELAY','DISTANCE']

categoricalColumns = ['OP_CARRIER']

cols = df.columns

# COMMAND ----------

# MAGIC %md ### MinMaxScaler

# COMMAND ----------

assembler = VectorAssembler(inputCols=numericCols, outputCol="Numfeatures")
df = assembler.transform(df)

Scaler = MinMaxScaler(inputCol="Numfeatures", outputCol="scaledFeatures")
df = Scaler.fit(df).transform(df)

# COMMAND ----------

# MAGIC %md ### CAT COLUMS

# COMMAND ----------

stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder (inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

# COMMAND ----------

# MAGIC %md ### LABELS

# COMMAND ----------

label_stringIdx = StringIndexer(inputCol = 'DEP_DEL15', outputCol = 'label')
stages += [label_stringIdx]

# COMMAND ----------

# MAGIC %md ### ASSEMBLER

# COMMAND ----------

assemblerInputs = [c + "classVec" for c in categoricalColumns] 
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="featuresCat")
stages += [assembler]

# COMMAND ----------

# MAGIC %md ### PIPELINE

# COMMAND ----------

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'featuresCat'] + cols + ['Numfeatures','scaledFeatures']
df = df.select(selectedCols)

# COMMAND ----------

# MAGIC %md ### JOIN SCALED VECTOR TO CAT VECTOR

# COMMAND ----------

assembler = VectorAssembler(inputCols=["featuresCat", "scaledFeatures"],outputCol="features")
df = assembler.transform(df)

# COMMAND ----------

# MAGIC %md ## TRAIN/TEST

# COMMAND ----------

if train_simple == 1:
  train, test = df.randomSplit([0.8, 0.2], seed = 1)
  print("Training Dataset Count: " + str(train.count()))
  print("Test Dataset Count: " + str(test.count()))

if train_simple == 2:  
  # Training set
  train = df.where((col('YEAR') == 2015) | (col('YEAR') == 2016) | (col('YEAR') == 2017) | (col('YEAR') == 2018) )
  print(f'{train.count():} flight records in training data')
  # Development set
  dev = df.where((col('YEAR') == 2019) & (col('MONTH')<7) )
  print(f'{dev.count():} flight records in development data')
  # Test set
  test = df.where((col('YEAR') == 2019) & (col('MONTH')>=7))
  print(f'{test.count():} flight records in test data')
  
if train_simple == 3:
  # TRAIN AND ALL DATA BUT LAST 3M
  train = df.where(((col('YEAR') == 2015) | (col('YEAR') == 2016) | (col('YEAR') == 2017) | (col('YEAR') == 2018)) | ((col('YEAR') == 2019) & (col('MONTH') <10) ))
  print(f'{train.count():} records in train data')

  # TEST/DEV ON LAST 3M OF DATA
  test, dev = (df.where((col('YEAR') == 2019) & (col('MONTH')>=10))).randomSplit([0.5,0.5],seed=1)
  print(f'{test.count():} records in test data')  
  print(f'{dev.count():} records in dev data')  

# COMMAND ----------

# MAGIC %md ### OVER/UNDER SAMPLE

# COMMAND ----------

train = sample_df(train,ratio,sample_flag)

# COMMAND ----------

# MAGIC %md # Models

# COMMAND ----------

# MAGIC %md ## LogisticRegression

# COMMAND ----------

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)

# COMMAND ----------

beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

# COMMAND ----------

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.show()
print('Training set area Under ROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# COMMAND ----------

predictions = lrModel.transform(test)
display(predictions[["label","prediction"]])

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

display(lrModel, train, "ROC")

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

# COMMAND ----------

# Print the coefficients and intercept for logistic regression
#print("Coefficients: " + str(lrModel.coefficients))
#print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

class_temp = predictions.select("label").groupBy("label").count().sort('count', ascending=False).toPandas()
class_temp = class_temp["label"].values.tolist()
class_names = map(str, class_temp)
print(class_names)

# COMMAND ----------

y_true = predictions.select("label")
y_true = y_true.toPandas()
y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()
cnf_matrix = confusion_matrix(y_true, y_pred)

# COMMAND ----------

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['no_delay', 'delay']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

# COMMAND ----------

# Plot normalized confusion matrix
plt.figure()
class_names = ['no_delay', 'delay']
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# COMMAND ----------

print(classification_report(y_true, y_pred, target_names=class_names))

# COMMAND ----------

print(accuracy_score(y_true, y_pred))

# COMMAND ----------

# MAGIC %md ##DecisionTree

# COMMAND ----------

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=6)
 
# Train model with Training Data
dtModel = dt.fit(train)

# COMMAND ----------

print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)

# COMMAND ----------

display(dtModel)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(test)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction")
display(selected)

# COMMAND ----------

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md ## RandomForestClassifier

# COMMAND ----------

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees = 500)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
display(predictions[["label","prediction"]])

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

ExtractFeatureImp(rfModel.featureImportances, train, "features").head(20)

# COMMAND ----------

# MAGIC %md ## GBTClassifier

# COMMAND ----------

gbt = GBTClassifier(maxIter=100)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)

# COMMAND ----------

display(predictions[["label","prediction"]])

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

# MAGIC %md ## Linear Support Vector Machine

# COMMAND ----------

lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvcModel = lsvc.fit(train)
predictions = lsvcModel.transform(test)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

# MAGIC %md ## Naive Bayes

# COMMAND ----------

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)
predictions = model.transform(test)

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------


