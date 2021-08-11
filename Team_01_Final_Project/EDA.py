# Databricks notebook source
# MAGIC %md # Flight Departure Delay Predictions
# MAGIC ## Exploratory Data Analysis (EDA)
# MAGIC 
# MAGIC ## Team members:
# MAGIC - Isabel Garcia Pietri
# MAGIC - Madhu Hegde
# MAGIC - Amit Karandikar
# MAGIC - Piotr Parkitny

# COMMAND ----------

# MAGIC %md ## EDA 
# MAGIC Exploratory data analysis will be conducted to review the data and make sure we don't make any assumptions that are not based on data. 
# MAGIC 
# MAGIC EDA tasks will help with identifying methods of joining the airlines data to the weather data. Big component of the EDA is to get a better understanding of patterns that can be used as features in our final solution. Outliers and anomalous events will need to be detected. Key is to find interesting relationships among features.
# MAGIC 
# MAGIC ### Overview:
# MAGIC The top EDA tasks that will help us make the decisions about how to implement the algorithm to be scalable:
# MAGIC - Identify outlier and abnormalities
# MAGIC - Use spark functionality and avoid any non-parallelizable libraries like pandas
# MAGIC 
# MAGIC ### Early Challenges:
# MAGIC Some early challenges that we have discovered are as follows:
# MAGIC - The dataset is not balanced, not delayed flights are 4 times bigger then delayed flights. This creates a problem where poorly choses classification algorithm might simply always predict the majority class
# MAGIC - Joining of the weather data to the airports needs to be done on Lat/Long along with the timestamp
# MAGIC - How to treat flights that do not land at the original destination
# MAGIC - Lots of columns appear to have no data or have bad data
# MAGIC - How to deal with cancellations
# MAGIC - Various fields have different scales and will need to be normalized
# MAGIC - Some fields have special characters whose meaning will need to be deducted 
# MAGIC 
# MAGIC ### Data Overview:
# MAGIC Descriptive statistics for each of the dataset’s tables provided:
# MAGIC - mean
# MAGIC - stddev
# MAGIC - min
# MAGIC - max 
# MAGIC - count 
# MAGIC 
# MAGIC The descriptive statistics highlighted possible outliers, missing values and the use of 9999 missing value identifier.
# MAGIC 
# MAGIC ### Documentation:
# MAGIC - Detail Explanation of all the fields can be found in the following document https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# MAGIC 
# MAGIC ### Outcome:
# MAGIC The outcome measure is DEP_DEL15 from the airlines table. 
# MAGIC 
# MAGIC ### Results:
# MAGIC EDA steps and further analysis are conducted by loading the data into Databricks Database. The top benefits are preventing Data Corruption, Faster Queries, Increase Data Freshness, Reproduce ML Models, Achieve Compliance, and ability to have ACID-compliant reads and writes.
# MAGIC 
# MAGIC ### Three Stage Approach
# MAGIC - First stage will be to work with the dataset that is composed of 1 Quarter of flight information for flights departing from two major US airports for the first quarter of 2015. Working with the smaller dataset will allow to quickly detect patterns without the need to work with the entire dataset which can be very slow.
# MAGIC - Second stage will be to use the full dataset, second stage can uncover long-term patterns.
# MAGIC - Third stage of the EDA is to check the processed data feature relationships to make sure they meet expectations.

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 1: EDA Raw Data
# MAGIC In the first part of the EDA will be using the sample data which is composed of 3 months of airlines data and filtered weather data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Package Imports

# COMMAND ----------

from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, expr, unix_timestamp, avg, round, lag, hour, to_date, row_number, explode, array, lit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from  pandas.plotting import lag_plot

from pyspark.mllib.stat import Statistics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import udf
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import PCA
from pyspark.ml.feature import MinMaxScaler

# COMMAND ----------

# MAGIC %md ### Define Function

# COMMAND ----------

def get_dtype(df,coltype):
  col_list = []
  for name,dtype in df.dtypes:
    if dtype == coltype:
      col_list.append(name)
  return col_list

# COMMAND ----------

# MAGIC %md ### Load Data

# COMMAND ----------

#Load Sample Set
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

# COMMAND ----------

#Check Duplicates
print(df_airlines.count())
df_airlines = df_airlines.dropDuplicates()
print(df_airlines.count())

print(df_weather.count())
df_weather = df_weather.dropDuplicates()
print(df_weather.count())

print(df_stations.count())
df_stations = df_stations.dropDuplicates()
print(df_stations.count())

# COMMAND ----------

#MAKE IT READY FOR SQL
df_airlines.createOrReplaceTempView("df_airlines")
df_weather.createOrReplaceTempView("df_weather")
df_stations.createOrReplaceTempView("df_stations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Data to DB
# MAGIC - First, we load the data into the Databricks DB. 
# MAGIC - Having the data inside the database allows a very easy method of sharing results among team members

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_stations_sql; CREATE TABLE df_stations_sql AS SELECT * FROM df_stations; 
# MAGIC DROP TABLE IF EXISTS df_airlines_sql; CREATE TABLE df_airlines_sql AS SELECT * FROM df_airlines; 
# MAGIC DROP TABLE IF EXISTS df_weather_sql; CREATE TABLE df_weather_sql AS SELECT * FROM df_weather;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Descriptive Statistics
# MAGIC - Create descriptive statistics for the loaded data
# MAGIC - Descriptive statistics provide basic information about the variables in the dataset and it might highlight protentional relationships between variables
# MAGIC - Results are loaded into pandas frame to allow for easy visibility

# COMMAND ----------

# CREATE DESCRIPTIVE STATISTICS
df_airlines_summary = spark.createDataFrame(df_airlines.describe().toPandas())
df_airlines_summary.createOrReplaceTempView("df_airlines_summary")

df_weather_summary = spark.createDataFrame(df_weather.describe().toPandas())
df_weather_summary.createOrReplaceTempView("df_weather_summary")

df_stations_summary = spark.createDataFrame(df_stations.describe().toPandas())
df_stations_summary.createOrReplaceTempView("df_stations_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Load Descriptive Statistics To DB
# MAGIC - Save descriptive statistics into DB.
# MAGIC - This allows everyone access to the tables.

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_airlines_summary_sql; CREATE TABLE df_airlines_summary_sql AS SELECT * FROM df_airlines_summary; 
# MAGIC DROP TABLE IF EXISTS df_weather_summary_sql; CREATE TABLE df_weather_summary_sql AS SELECT * FROM df_weather_summary;
# MAGIC DROP TABLE IF EXISTS df_stations_summary_sql; CREATE TABLE df_stations_summary_sql AS SELECT * FROM df_stations_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Review Descriptive Statistics
# MAGIC Results for descriptive statistics
# MAGIC 
# MAGIC 
# MAGIC ### table: airlines
# MAGIC - We are working with a sample of the data
# MAGIC - Row Count = 161057
# MAGIC - Large number of features are null. Example: DIV4_WHEELS_ON
# MAGIC - We have a mixture of feature types
# MAGIC - We have panel data
# MAGIC - Will need to address cancellations as they might require special treatment
# MAGIC 
# MAGIC 
# MAGIC ### table: weather
# MAGIC - We are working with a sample of the data
# MAGIC - Row Count = 29643209
# MAGIC - Some columns will require special treatment: split by comma
# MAGIC - Fields contain special characters (+/-, N etc..)
# MAGIC - We have panel data
# MAGIC 
# MAGIC 
# MAGIC ### table: stations
# MAGIC - This is a dimension table
# MAGIC - Table contains stations from around the world, will need to only keep US Stations

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from df_airlines_summary_sql;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from df_weather_summary_sql;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from df_stations_summary_sql;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Review Raw Data
# MAGIC - Quick View at the top two records for each data source
# MAGIC 
# MAGIC ### table: airlines
# MAGIC - We are working with a sample of the data
# MAGIC - Data is provided by TranStats data collection available from the U.S. Department of Transportation 
# MAGIC 
# MAGIC ### table: weather
# MAGIC - We are working with a sample of the data
# MAGIC - Data is from the National Oceanic and Atmospheric Administration repository
# MAGIC 
# MAGIC ### table: stations
# MAGIC - We are working with the entire data set
# MAGIC - The table includes stations from around the world
# MAGIC - In order to speed up analysis and joins as first step we should only keep US stations

# COMMAND ----------

# MAGIC %sql SELECT * FROM df_airlines LIMIT 2

# COMMAND ----------

# MAGIC %sql SELECT * FROM df_weather LIMIT 2

# COMMAND ----------

# MAGIC %sql SELECT * FROM df_stations_sql LIMIT 2

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 7: Dataset Balance
# MAGIC - The dataset is not balanced
# MAGIC - Non delayed flights are ~ 3 times bigger then delayed flights
# MAGIC - Potential problem for most classification algorithms as they are designed on the assumption of an equal number of observations in each class
# MAGIC - The balance will also need to be checked in the final dataset by either resampling the training set or using a SMOTE method.

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   'Delay >= 15 min' DelayType, 
# MAGIC    count(*) Flight_CNT
# MAGIC FROM df_airlines_sql
# MAGIC WHERE DEP_DEL15 = 1
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Delay < 15 min' Delay_Type, 
# MAGIC    count(*) Flight_CNT
# MAGIC FROM df_airlines_sql
# MAGIC WHERE DEP_DEL15 = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Delay Length
# MAGIC - Majority of the delays are under 30 minutes
# MAGIC - However, some extreme delays due exists where the delay is more than 1000 minutes, histogram has a very long tail
# MAGIC - Small group of flights leaves early

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   DEP_DELAY
# MAGIC FROM df_airlines_sql

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Delays by Destination
# MAGIC - A heatmap with the date on the x-axis allows to easily see any patterns that are time based.
# MAGIC - Delays by destination overall seem random over time
# MAGIC - As we only have three months of data it is hard to make any conclusion from the heatmap

# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
  A.Date date,
  A.var,
  (B.cnt / cast(A.cnt as float)) * 100 as cnt
FROM 
    (
    SELECT 
      last_day(AN.FL_DATE) date, 
      DEST var, 
      count(*) cnt
    FROM df_airlines_sql AN
    GROUP BY last_day(AN.FL_DATE), AN.DEST
  ) A
  LEFT JOIN
  (
  SELECT 
      last_day(BN.FL_DATE) date, 
      DEST var, 
      count(*) cnt
    FROM df_airlines_sql BN
    WHERE BN.DEP_DEL15  = 1
    GROUP BY last_day(BN.FL_DATE), BN.DEST   
  ) B
ON A.date = B.date AND A.var = B.var
""")

DATA_DF = df_temp.toPandas()
DATA_DF_GB = DATA_DF.groupby(by=["date"]).sum()
DATA_DF_GB.reset_index(inplace=True)

col_join = ['date']
return_df = pd.merge(DATA_DF, DATA_DF_GB, left_on=col_join, right_on=col_join)
return_df["cnt"] = (return_df["cnt_x"] / return_df["cnt_y"]) * 100

DATA_DFP = return_df.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=1)
fig = plt.figure(figsize=(5, 40),frameon =True, dpi=200)   
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.2f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Destination Airport',title=("Delays by Destination"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Delays By Departure Time
# MAGIC - Delays appear to happen later in the day, most likely as the day progresses the chances are higher that your airplane was delayed on another trip
# MAGIC - The trend is consistent across the three sample months

# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
  A.Date date,
  A.var,
  (B.cnt / cast(A.cnt as float)) * 100 as cnt
FROM 
    (
    SELECT 
      last_day(AN.FL_DATE) date, 
      lpad(cast(cast(AN.DEP_TIME/100 as int) as string),2,'0') var, 
      count(*) cnt
    FROM df_airlines_sql AN
    GROUP BY last_day(AN.FL_DATE), lpad(cast(cast(AN.DEP_TIME/100 as int) as string),2,'0')
  ) A
  LEFT JOIN
  (
  SELECT 
      last_day(BN.FL_DATE) date, 
      lpad(cast(cast(BN.DEP_TIME/100 as int) as string),2,'0') var, 
      count(*) cnt
    FROM df_airlines_sql BN
    WHERE BN.DEP_DEL15 = 1
    GROUP BY last_day(BN.FL_DATE), lpad(cast(cast(BN.DEP_TIME/100 as int) as string),2,'0')    
  ) B
ON A.date = B.date AND A.var = B.var
WHERE A.var is not null
""")

DATA_DF = df_temp.toPandas()
DATA_DF_GB = DATA_DF.groupby(by=["date"]).sum()
DATA_DF_GB.reset_index(inplace=True)

col_join = ['date']
return_df = pd.merge(DATA_DF, DATA_DF_GB, left_on=col_join, right_on=col_join)
return_df["cnt"] = (return_df["cnt_x"] / return_df["cnt_y"]) * 100

DATA_DFP = return_df.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=1.4)
fig = plt.figure(figsize=(3, 10),frameon =True, dpi=200)  
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.1f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Hour of Day',title=("% of Delays by Hour of Day"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Delays By Operator
# MAGIC - Delays occur more/less for certain operators
# MAGIC - Operator B9 appears to have consistent delays

# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
  A.Date date,
  A.OP_CARRIER var,
  (B.cnt / cast(A.cnt as float)) * 100 as cnt
FROM 
    (
    SELECT 
      last_day(AN.FL_DATE) date, 
      AN.OP_CARRIER, 
      count(*) cnt
    FROM df_airlines_sql AN
    GROUP BY last_day(AN.FL_DATE), AN.OP_CARRIER
  ) A
  LEFT JOIN
  (
  SELECT 
      last_day(BN.FL_DATE) date, 
      BN.OP_CARRIER, 
      count(*) cnt
    FROM df_airlines_sql BN
    WHERE BN.DEP_DELAY >= 15
    GROUP BY last_day(BN.FL_DATE), BN.OP_CARRIER    
  ) B
ON A.date = B.Date AND A.OP_CARRIER = B.OP_CARRIER
""")

DATA_DF = df_temp.toPandas()
DATA_DF_GB = DATA_DF.groupby(by=["date"]).sum()
DATA_DF_GB.reset_index(inplace=True)

col_join = ['date']
return_df = pd.merge(DATA_DF, DATA_DF_GB, left_on=col_join, right_on=col_join)
return_df["cnt"] = (return_df["cnt_x"] / return_df["cnt_y"]) * 100

DATA_DFP = return_df.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=1.4)
fig = plt.figure(figsize=(3, 10),frameon =True, dpi=200)     
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.1f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Operator',title=("% of Delays by Operator"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Top Destination
# MAGIC - Destination distribution looks very smooth
# MAGIC - LGA is the largest airport

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   DEST Destination, 
# MAGIC   count(*) Count
# MAGIC FROM df_airlines_sql
# MAGIC GROUP BY DEST
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Top Origin
# MAGIC - Due to only having sample data we only have two origins

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   ORIGIN, 
# MAGIC   count(*) Count
# MAGIC FROM df_airlines_sql
# MAGIC GROUP BY ORIGIN
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14: Correlation Airlines
# MAGIC 
# MAGIC Here a correlation heatmap is created to quickly visualize if any linear relationship exists among the features. We find that:
# MAGIC - Some features are correlated and will need to be removed
# MAGIC - Many correlated features, 
# MAGIC - Nothing strongly correlated to delay label

# COMMAND ----------

df_airlines_clean = sqlContext.sql(""" SELECT * FROM df_airlines_sql""")
df_airlines_clean = df_airlines_clean.drop("DIV4_WHEELS_OFF",
"DIV4_TAIL_NUM","DIV5_AIRPORT","DIV5_AIRPORT_ID","DIV5_AIRPORT_SEQ_ID","DIV5_WHEELS_ON","DIV5_TOTAL_GTIME","DIV5_LONGEST_GTIME","DIV5_WHEELS_OFF",
"DIV5_TAIL_NUM","DIV3_WHEELS_OFF","DIV3_TAIL_NUM","DIV4_AIRPORT","DIV4_AIRPORT_ID","DIV4_AIRPORT_SEQ_ID","DIV4_WHEELS_ON","DIV4_TOTAL_GTIME",
"DIV4_LONGEST_GTIME","DIV4_WHEELS_OFF","DIV3_WHEELS_OFF","DIV3_TAIL_NUM","DIV4_AIRPORT","DIV4_AIRPORT_ID","DIV4_AIRPORT_SEQ_ID","DIV4_WHEELS_ON",
"DIV4_TOTAL_GTIME","DIV4_LONGEST_GTIME","DIV2_TOTAL_GTIME","DIV2_LONGEST_GTIME","DIV2_WHEELS_OFF","DIV2_TAIL_NUM","DIV3_AIRPORT","DIV3_AIRPORT_ID",
"DIV3_AIRPORT_SEQ_ID","DIV3_WHEELS_ON","DIV3_TOTAL_GTIME","DIV3_LONGEST_GTIME","DIV2_TOTAL_GTIME","DIV2_LONGEST_GTIME","DIV2_WHEELS_OFF","DIV2_TAIL_NUM",
"DIV3_AIRPORT","DIV3_AIRPORT_ID","DIV3_AIRPORT_SEQ_ID","DIV3_WHEELS_ON","DIV3_TOTAL_GTIME","DIV3_LONGEST_GTIME","DIV1_LONGEST_GTIME","DIV1_WHEELS_OFF",
"DIV1_TAIL_NUM","DIV2_AIRPORT","DIV2_AIRPORT_ID","DIV2_AIRPORT_SEQ_ID","DIV1_AIRPORT_SEQ_ID","DIV2_WHEELS_ON","DIV2_TOTAL_GTIME","DIV2_LONGEST_GTIME",
"SECURITY_DELAY","DIV_REACHED_DEST","YEAR","MONTH","QUARTER","DAY_OF_MONTH","DAY_OF_WEEK","DIV_AIRPORT_LANDINGS","DIV1_AIRPORT_ID","DIV_1_AIRPORT_SEQ_ID",
"DIV1_WHEELS_ON","FLIGHTS","DIV_ACTUAL_ELAPSED_TIME","DIV_DISTANCE","DIV_ARR_DELAY","DIV1_TOTAL_GTIME"
)
df_airlines_clean = df_airlines_clean.na.fill(0)

df = df_airlines_clean
inputCols=df.columns
inputCols=get_dtype(df,"int") + get_dtype(df,"double")

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=inputCols, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
result = matrix.collect()[0]["pearson({})".format(vector_col)].values

resh_results = result.reshape(int(math.sqrt(len(result))),int(math.sqrt(len(result))))

sns.set(font_scale=1.2)
ax = plt.figure(figsize=(28,28), frameon=True, dpi=200)
ax = sns.heatmap(resh_results,annot=True,xticklabels=inputCols, yticklabels=inputCols,cbar=0,annot_kws={"size": 13},fmt='0.1f',cmap="Blues")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 15: Weather Values Extract
# MAGIC 
# MAGIC The data is split into columns so a correlation analysis can be used to determine if any linear correlation exists in the weather data. 

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_clean_sql;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_sql AS
# MAGIC SELECT
# MAGIC    STATION
# MAGIC   ,DATE
# MAGIC   ,to_timestamp(concat(YEAR(DATE),'-',MONTH(DATE),'-',DAY(DATE),'T',HOUR(DATE),':',floor(minute(DATE)/10)*10,':00.000+0000')) AS DATE_JOIN_MN
# MAGIC   ,to_timestamp(concat(YEAR(DATE),'-',MONTH(DATE),'-',DAY(DATE),'T',HOUR(DATE),':','00',':00.000+0000')) AS DATE_JOIN_HH
# MAGIC   ,to_timestamp(concat(YEAR(DATE),'-',MONTH(DATE),'-',DAY(DATE),'T','00',':','00',':00.000+0000')) AS DATE_JOIN_DA
# MAGIC   
# MAGIC   ,SOURCE
# MAGIC   ,cast(LATITUDE  as float) as LATITUDE
# MAGIC   ,cast(LONGITUDE as float) as LONGITUDE
# MAGIC   ,cast(ELEVATION as float) as ELEVATION
# MAGIC   
# MAGIC   ,SPLIT(NAME,',')[0] as NAME_0
# MAGIC   ,SPLIT(NAME,',')[1] as NAME_1
# MAGIC   
# MAGIC   ,REPORT_TYPE
# MAGIC   ,CALL_SIGN
# MAGIC   ,QUALITY_CONTROL
# MAGIC   
# MAGIC   ,WND
# MAGIC   ,cast(SPLIT(WND,',')[0] as int) as WND_0
# MAGIC   ,cast(SPLIT(WND,',')[1] as int) as WND_1
# MAGIC   ,SPLIT(WND,',')[2]              as WND_2
# MAGIC   ,cast(SPLIT(WND,',')[3] as int) as WND_3
# MAGIC   ,cast(SPLIT(WND,',')[4] as int) as WND_4
# MAGIC   
# MAGIC   ,CIG
# MAGIC   ,cast(SPLIT(CIG,',')[0] as int) as CIG_0
# MAGIC   ,cast(SPLIT(CIG,',')[1] as int) as CIG_1
# MAGIC   ,SPLIT(CIG,',')[2]              as CIG_2
# MAGIC   ,SPLIT(CIG,',')[3]              as CIG_3
# MAGIC   
# MAGIC   ,VIS
# MAGIC   ,cast(SPLIT(VIS,',')[0] as int) as VIS_0
# MAGIC   ,cast(SPLIT(VIS,',')[1] as int) as VIS_1
# MAGIC   ,cast(SPLIT(VIS,',')[2] as int) as VIS_2
# MAGIC   ,cast(SPLIT(VIS,',')[3] as int) as VIS_3
# MAGIC   
# MAGIC   ,TMP
# MAGIC   ,IF(substring(TMP,0,1) = '-', (cast(substring(SPLIT(TMP,',')[0],2) as int) * -1), (cast(substring(SPLIT(TMP,',')[0],2) as int))) as TMP_0
# MAGIC   ,cast( SPLIT(TMP,',')[1] as int)              as TMP_1
# MAGIC   
# MAGIC   ,DEW  
# MAGIC   ,IF(substring(DEW,0,1) = '-', (cast(substring(SPLIT(DEW,',')[0],2) as int) * -1), (cast(substring(SPLIT(DEW,',')[0],2) as int))) as DEW_0  
# MAGIC   ,cast( SPLIT(DEW,',')[1] as int)              as DEW_1
# MAGIC 
# MAGIC   ,SLP
# MAGIC   ,cast( SPLIT(SLP,',')[0] as int)              as SLP_0
# MAGIC   ,cast( SPLIT(SLP,',')[1] as int)              as SLP_1
# MAGIC 
# MAGIC   ,AW1
# MAGIC   
# MAGIC   ,GA1
# MAGIC   ,cast( SPLIT(GA1,',')[0] as int)              as GA1_0
# MAGIC   ,cast( SPLIT(GA1,',')[1] as int)              as GA1_1    
# MAGIC   ,IF(substring(SPLIT(GA1,',')[2],0,1) = '-', (cast(substring(SPLIT(GA1,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA1,',')[2],2) as int))) as GA1_2  
# MAGIC   ,cast( SPLIT(GA1,',')[3] as int)              as GA1_3  
# MAGIC   ,cast( SPLIT(GA1,',')[4] as int)              as GA1_4  
# MAGIC   ,cast( SPLIT(GA1,',')[5] as int)              as GA1_5  
# MAGIC     
# MAGIC   ,GA2
# MAGIC   ,cast( SPLIT(GA2,',')[0] as int)              as GA2_0
# MAGIC   ,cast( SPLIT(GA2,',')[1] as int)              as GA2_1    
# MAGIC   ,IF(substring(SPLIT(GA2,',')[2],0,1) = '-', (cast(substring(SPLIT(GA2,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA2,',')[2],2) as int))) as GA2_2  
# MAGIC   ,cast( SPLIT(GA2,',')[3] as int)              as GA2_3  
# MAGIC   ,cast( SPLIT(GA2,',')[4] as int)              as GA2_4  
# MAGIC   ,cast( SPLIT(GA2,',')[5] as int)              as GA2_5  
# MAGIC   
# MAGIC   ,GA3
# MAGIC   ,cast( SPLIT(GA3,',')[0] as int)              as GA3_0
# MAGIC   ,cast( SPLIT(GA3,',')[1] as int)              as GA3_1    
# MAGIC   ,IF(substring(SPLIT(GA3,',')[2],0,1) = '-', (cast(substring(SPLIT(GA3,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA3,',')[2],2) as int))) as GA3_2    
# MAGIC   ,cast( SPLIT(GA3,',')[3] as int)              as GA3_3  
# MAGIC   ,cast( SPLIT(GA3,',')[4] as int)              as GA3_4  
# MAGIC   ,cast( SPLIT(GA3,',')[5] as int)              as GA3_5  
# MAGIC   
# MAGIC   ,GA4
# MAGIC   ,cast( SPLIT(GA4,',')[0] as int)              as GA4_0
# MAGIC   ,cast( SPLIT(GA4,',')[1] as int)              as GA4_1    
# MAGIC   ,IF(substring(SPLIT(GA4,',')[2],0,1) = '-', (cast(substring(SPLIT(GA4,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA4,',')[2],2) as int))) as GA4_2    
# MAGIC   ,cast( SPLIT(GA4,',')[3] as int)              as GA4_3  
# MAGIC   ,cast( SPLIT(GA4,',')[4] as int)              as GA4_4  
# MAGIC   ,cast( SPLIT(GA4,',')[5] as int)              as GA4_5  
# MAGIC   
# MAGIC   ,GE1
# MAGIC   ,cast( SPLIT(GE1,',')[0] as int)              as GE1_0
# MAGIC   ,SPLIT(GE1,',')[1]                            as GE1_1    
# MAGIC   ,IF(substring(SPLIT(GE1,',')[2],0,1) = '-', (cast(substring(SPLIT(GE1,',')[2],2) as int) * -1), (cast(substring(SPLIT(GE1,',')[2],2) as int))) as GE1_2    
# MAGIC   ,IF(substring(SPLIT(GE1,',')[3],0,1) = '-', (cast(substring(SPLIT(GE1,',')[3],2) as int) * -1), (cast(substring(SPLIT(GE1,',')[3],2) as int))) as GE1_3  
# MAGIC   
# MAGIC   ,GF1
# MAGIC   ,cast( SPLIT(GF1,',')[0] as int)              as GF1_0
# MAGIC   ,cast( SPLIT(GF1,',')[1] as int)              as GF1_1
# MAGIC   ,cast( SPLIT(GF1,',')[2] as int)              as GF1_2
# MAGIC   ,cast( SPLIT(GF1,',')[3] as int)              as GF1_3
# MAGIC   ,cast( SPLIT(GF1,',')[4] as int)              as GF1_4
# MAGIC   ,cast( SPLIT(GF1,',')[5] as int)              as GF1_5
# MAGIC   ,cast( SPLIT(GF1,',')[6] as int)              as GF1_6
# MAGIC   ,cast( SPLIT(GF1,',')[7] as int)              as GF1_7
# MAGIC   ,cast( SPLIT(GF1,',')[8] as int)              as GF1_8
# MAGIC   ,cast( SPLIT(GF1,',')[9] as int)              as GF1_9
# MAGIC   ,cast( SPLIT(GF1,',')[10] as int)             as GF1_10
# MAGIC   ,cast( SPLIT(GF1,',')[11] as int)             as GF1_11
# MAGIC   ,cast( SPLIT(GF1,',')[12] as int)             as GF1_12
# MAGIC    
# MAGIC   ,KA1
# MAGIC   ,cast( SPLIT(KA1,',')[0] as int)              as KA1_0
# MAGIC   ,SPLIT(KA1,',')[1]                            as KA1_1
# MAGIC   ,IF(substring(SPLIT(KA1,',')[2],0,1) = '-', (cast(substring(SPLIT(KA1,',')[2],2) as int) * -1), (cast(substring(SPLIT(KA1,',')[2],2) as int))) as KA1_2B  
# MAGIC   ,cast( SPLIT(KA1,',')[3] as int)              as KA1_3 
# MAGIC FROM df_weather_sql
# MAGIC WHERE STATION IS NOT NULL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 16: Weather Correlation
# MAGIC - Some features are correlated and will need to be removed
# MAGIC - Many correlated features, 
# MAGIC - weather table features have high correlation and some will need to be removed

# COMMAND ----------

df_weather_clean = sqlContext.sql(""" SELECT * FROM df_weather_clean_sql""")
df_weather_clean = df_weather_clean.na.fill(0)

df = df_weather_clean
inputCols=df.columns
inputCols=get_dtype(df,"int") + get_dtype(df,"double")

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=inputCols, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
result = matrix.collect()[0]["pearson({})".format(vector_col)].values

resh_results = result.reshape(int(math.sqrt(len(result))),int(math.sqrt(len(result))))

sns.set(font_scale=1.2)
ax = plt.figure(figsize=(28,28), frameon=True, dpi=200)
ax = sns.heatmap(resh_results,annot=True,xticklabels=inputCols, yticklabels=inputCols,cbar=0,annot_kws={"size": 13},fmt='0.1f',cmap="Blues")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 17: Delays by Day Of Week
# MAGIC - Delays are less likely to occur on day of week = 7

# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
  A.Date date,
  A.var var,
  (B.cnt / cast(A.cnt as float)) * 100 as cnt
FROM 
    (
    SELECT 
      last_day(AN.FL_DATE) date, 
      dayofweek(AN.FL_DATE) var, 
      count(*) cnt
    FROM df_airlines_sql AN
    GROUP BY last_day(AN.FL_DATE), dayofweek(AN.FL_DATE)
  ) A
  LEFT JOIN
  (
  SELECT 
      last_day(BN.FL_DATE) date, 
      dayofweek(BN.FL_DATE) var, 
      count(*) cnt
    FROM df_airlines_sql BN
    WHERE BN.DEP_DELAY >= 15
    GROUP BY last_day(BN.FL_DATE), dayofweek(BN.FL_DATE)
  ) B
ON A.date = B.date AND A.var = B.var
""")

DATA_DF = df_temp.toPandas()
DATA_DFP = DATA_DF.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=1)
fig = plt.figure(figsize=(10, 10),frameon =True)   
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.2f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Day of Week',title=("% of Delays by Day of Week"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 18: Delays by Trip Group
# MAGIC - It is not easy to detect pattern based on only three months of data
# MAGIC - Group 5-6 has higher delays across three months
# MAGIC - Group 9 is constantly the lowest

# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
  A.Date date,
  A.var var,
  (B.cnt / cast(A.cnt as float)) * 100 as cnt
FROM 
    (
    SELECT 
      last_day(AN.FL_DATE) date, 
      AN.DISTANCE_GROUP var, 
      count(*) cnt
    FROM df_airlines_sql AN
    GROUP BY last_day(AN.FL_DATE), AN.DISTANCE_GROUP
  ) A
  LEFT JOIN
  (
  SELECT 
      last_day(BN.FL_DATE) date, 
      BN.DISTANCE_GROUP var, 
      count(*) cnt
    FROM df_airlines_sql BN
    WHERE BN.DEP_DELAY >= 15
    GROUP BY last_day(BN.FL_DATE), BN.DISTANCE_GROUP
  ) B
ON A.date = B.date AND A.var = B.var
""")

DATA_DF = df_temp.toPandas()
DATA_DFP = DATA_DF.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=1)
fig = plt.figure(figsize=(10, 10),frameon =True)   
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.2f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Distance Group',title=("Delay by Distance Group"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 19: NULL Check

# COMMAND ----------

# check for null values 
display(df_airlines.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_airlines.columns]).toPandas())
display(df_stations.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_stations.columns]).toPandas())

# COMMAND ----------

cols = ['STATION', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL',
 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AW1', 'GA1', 'GA2', 'GA3', 'GA4', 'GE1', 'GF1', 'KA1', 'KA2', 'MA1', 'MD1', 'MW1', 'MW2', 'OC1', 'OD1', 'OD2', 'REM', 'EQD', 'AW2', 'AX4', 'GD1',
 'AW5', 'GN1', 'AJ1', 'AW3', 'MK1', 'KA4', 'GG3', 'AN1', 'RH1', 'AU5', 'HL1', 'OB1', 'AT8', 'AW7', 'AZ1', 'CH1', 'RH3', 'GK1', 'IB1', 'AX1', 'CT1', 'AK1', 'CN2', 'OE1', 'MW5', 'AO1', 'KA3',
 'AA3', 'CR1', 'CF2', 'KB2', 'GM1', 'AT5', 'AY2', 'MW6', 'MG1', 'AH6', 'AU2', 'GD2', 'AW4', 'MF1', 'AA1', 'AH2', 'AH3', 'OE3', 'AT6', 'AL2', 'AL3', 'AX5', 'IB2', 'AI3', 'CV3', 'WA1', 'GH1',
 'KF1', 'CU2', 'CT3', 'SA1', 'AU1', 'KD2', 'AI5', 'GO1', 'GD3', 'CG3', 'AI1', 'AL1', 'AW6', 'MW4', 'AX6', 'CV1', 'ME1', 'KC2', 'CN1', 'UA1', 'GD5', 'UG2', 'AT3', 'AT4', 'GJ1', 'MV1', 'GA5',
 'CT2', 'CG2', 'ED1', 'AE1', 'CO1', 'KE1', 'KB1', 'AI4', 'MW3', 'KG2', 'AA2', 'AX2', 'AY1', 'RH2', 'OE2', 'CU3', 'MH1', 'AM1', 'AU4', 'GA6', 'KG1', 'AU3', 'AT7', 'KD1', 'GL1', 'IA1', 'GG2',
 'OD3', 'UG1', 'CB1', 'AI6', 'CI1', 'CV2', 'AZ2', 'AD1', 'AH1', 'WD1', 'AA4', 'KC1', 'IA2', 'CF3', 'AI2', 'AT1', 'GD4', 'AX3', 'AH4', 'KB3', 'CU1', 'CN4', 'AT2', 'CG1', 'CF1', 'GG1', 'MV2',
 'CW1', 'GG4', 'AB1', 'AH5', 'CN3']

# COMMAND ----------

df_weather.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in cols]).toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 20: Weather Data - Report Type
# MAGIC First let's check what type of data we have. Based on the https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf we will remove
# MAGIC - SOD – Summary of Day. A summary of the weather observations for the previous 24 hours
# MAGIC - SOM – Summary of Month. A summary of the weather observations for the previous month

# COMMAND ----------

# MAGIC %sql
# MAGIC select report_type, 
# MAGIC count(*) cnt
# MAGIC from df_weather
# MAGIC group by report_type
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 21: Weather Data - Quality Control

# COMMAND ----------

# MAGIC %sql
# MAGIC select QUALITY_CONTROL, 
# MAGIC count(*) cnt
# MAGIC from df_weather
# MAGIC group by report_type
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %md ##Step 22: Checking temporal aspects of delays

# COMMAND ----------

lp_df = sqlContext.sql("""SELECT * FROM airlines_with_features_sql""")
lp_df = lp_df.where((col('DEP_DELAY').isNotNull()) & (col('ORIGIN') == 'ORD') | (col('ORIGIN') == 'ATL')).orderBy('FL_DATE', 'CRS_DEP_TIME').toPandas()

# print(lp_df.columns)
# df1 = df.filter(col('ORIGIN') == 1)
# df1 = df1.toPandas()
plt.figure()
lag_plot(lp_df['DEP_DELAY'], lag=2)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC # PART 2: EDA On Processed Data

# COMMAND ----------

df_airlines = spark.read.parquet("/mnt/Azure/processed_data/airlines_processed.parquet")
df_weather = spark.read.parquet("/mnt/Azure/processed_data/weather_processed.parquet")
airport_dst_hh_cnt = spark.read.parquet("/mnt/Azure/processed_data/airport_dst_hh_processed.parquet/")
airport_src_hh_cnt = spark.read.parquet("/mnt/Azure/processed_data/airport_src_hh_processed.parquet/")

# COMMAND ----------

#Check Duplicates
print(df_airlines.count())
df_airlines = df_airlines.dropDuplicates()
print(df_airlines.count())

print(df_weather.count())
df_weather = df_weather.dropDuplicates()
print(df_weather.count())

print(airport_dst_hh_cnt.count())
airport_dst_hh_cnt = airport_dst_hh_cnt.dropDuplicates()
print(airport_dst_hh_cnt.count())

print(airport_src_hh_cnt.count())
airport_src_hh_cnt = airport_src_hh_cnt.dropDuplicates()
print(airport_src_hh_cnt.count())

# COMMAND ----------

df_airlines.createOrReplaceTempView("df_airlines")
df_weather.createOrReplaceTempView("df_weather")
airport_dst_hh_cnt.createOrReplaceTempView("airport_dst_hh_cnt")
airport_src_hh_cnt.createOrReplaceTempView("airport_src_hh_cnt")

# COMMAND ----------

#sqlContext.setConf("spark.driver.maxResultsSize","10g")

# COMMAND ----------

df_final = sqlContext.sql(""" 

Select AL1.*, 

--Source
AL2.STATION As Src_STATION,
AL2.DATE_JOIN_2H As Src_DATE_JOIN_2H,
AL2.WND_0 As Src_WND_0,
AL2.WND_1 As Src_WND_1,
AL2.WND_3 As Src_WND_3,
AL2.WND_4 As Src_WND_4,
AL2.CIG_0 As Src_CIG_0,
AL2.CIG_1 As Src_CIG_1,
AL2.VIS_0 As Src_VIS_0,
AL2.VIS_1 As Src_VIS_1,
AL2.VIS_2 As Src_VIS_2,
AL2.VIS_3 As Src_VIS_3,
AL2.TMP_0 As Src_TMP_0,
AL2.TMP_1 As Src_TMP_1,
AL2.DEW_0 As Src_DEW_0,
AL2.DEW_1 As Src_DEW_1,
AL2.SLP_0 As Src_SLP_0,
AL2.SLP_1 As Src_SLP_1,
AL2.GA1_0 As Src_GA1_0,
AL2.GA1_1 As Src_GA1_1,
AL2.GA1_2 As Src_GA1_2,
AL2.GA1_3 As Src_GA1_3,
AL2.GA1_4 As Src_GA1_4,
AL2.GA1_5 As Src_GA1_5,

--Destination
AL3.STATION As Dst_STATION,
AL3.DATE_JOIN_2H As Dst_DATE_JOIN_2H,
AL3.WND_0 As Dst_WND_0,
AL3.WND_1 As Dst_WND_1,
AL3.WND_3 As Dst_WND_3,
AL3.WND_4 As Dst_WND_4,
AL3.CIG_0 As Dst_CIG_0,
AL3.CIG_1 As Dst_CIG_1,
AL3.VIS_0 As Dst_VIS_0,
AL3.VIS_1 As Dst_VIS_1,
AL3.VIS_2 As Dst_VIS_2,
AL3.VIS_3 As Dst_VIS_3,
AL3.TMP_0 As Dst_TMP_0,
AL3.TMP_1 As Dst_TMP_1,
AL3.DEW_0 As Dst_DEW_0,
AL3.DEW_1 As Dst_DEW_1,
AL3.SLP_0 As Dst_SLP_0,
AL3.SLP_1 As Dst_SLP_1,
AL3.GA1_0 As Dst_GA1_0,
AL3.GA1_1 As Dst_GA1_1,
AL3.GA1_2 As Dst_GA1_2,
AL3.GA1_3 As Dst_GA1_3,
AL3.GA1_4 As Dst_GA1_4,
AL3.GA1_5 As Dst_GA1_5,

--Airport Usage 
AL4.DST_CNT,
AL5.DEP_CNT

from df_airlines AL1
LEFT JOIN df_weather AL2 ON 
      AL1.station_origin = AL2.STATION 
  AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL2.DATE_JOIN_2H
  
LEFT JOIN df_weather AL3 ON 
      AL1.station_dest = AL3.STATION 
  AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL3.DATE_JOIN_2H

LEFT JOIN airport_dst_hh_cnt AL4 ON AL1.ORIGIN_AIRPORT_ID = AL4.DEST_AIRPORT_ID
  AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL4.DATE_JOIN_HH
  
LEFT JOIN airport_src_hh_cnt AL5 ON AL1.DEST_AIRPORT_ID = AL5.ORIGIN_AIRPORT_ID
  AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL5.DATE_JOIN_HH

""")

# COMMAND ----------

df_final.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC Select count(*) from df_airlines AL1
# MAGIC LEFT JOIN df_weather AL2 ON 
# MAGIC       AL1.station_origin = AL2.STATION 
# MAGIC    AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL2.DATE_JOIN_2H
# MAGIC WHERE AL2.STATION IS NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC Select count(*) from df_airlines AL1
# MAGIC LEFT JOIN df_weather AL2 ON 
# MAGIC       AL1.station_origin = AL2.STATION 
# MAGIC     AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL2.DATE_JOIN_2H
# MAGIC WHERE AL2.STATION IS NOT NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC Select count(*) from df_airlines AL1
# MAGIC LEFT JOIN df_weather AL2 ON 
# MAGIC       AL1.station_origin = AL2.STATION 
# MAGIC     AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL2.DATE_JOIN_2H

# COMMAND ----------

df_final.createOrReplaceTempView("df_final")

# COMMAND ----------

# MAGIC %md ### Correlation

# COMMAND ----------

df = df_final.na.fill(0)

inputCols=df.columns
inputCols=get_dtype(df,"int") + get_dtype(df,"double") + get_dtype(df,"long")

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=inputCols, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
result = matrix.collect()[0]["pearson({})".format(vector_col)].values

resh_results = result.reshape(int(math.sqrt(len(result))),int(math.sqrt(len(result))))

sns.set(font_scale=1.1)
ax = plt.figure(figsize=(28,28), frameon=True, dpi=200)
ax = sns.heatmap(resh_results,annot=True,xticklabels=inputCols, yticklabels=inputCols,cbar=0,annot_kws={"size": 10},fmt='0.1f',cmap="Blues")

# COMMAND ----------

# MAGIC %md # PART 3: EDA On Full Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Data
# MAGIC - Full dataset is loaded into dataframes
# MAGIC - Duplication is checked, we find that airlines data is duplicated. ETL process will remove the duplication

# COMMAND ----------

#Full Set
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

# COMMAND ----------

#Check Duplicates
print(df_airlines.count())
df_airlines = df_airlines.dropDuplicates()
print(df_airlines.count())

print(df_weather.count())
df_weather = df_weather.dropDuplicates()
print(df_weather.count())

# COMMAND ----------

df_airlines.createOrReplaceTempView("df_airlines")
df_weather.createOrReplaceTempView("df_weather")
df_stations.createOrReplaceTempView("df_stations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Data to DB
# MAGIC - Loading data into the DB
# MAGIC - The top benefits of using the data lake: prevent Data Corruption, Faster Queries, Increase Data Freshness, Reproduce ML Models, Achieve Compliance and ability to have ACID-compliant reads and writes.

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_stations_prd_sql; CREATE TABLE df_stations_prd_sql AS SELECT * FROM df_stations; 
# MAGIC DROP TABLE IF EXISTS df_airlines_prd_sql; CREATE TABLE df_airlines_prd_sql AS SELECT * FROM df_airlines; 
# MAGIC DROP TABLE IF EXISTS df_weather_prd_sql; CREATE TABLE df_weather_prd_sql AS SELECT * FROM df_weather;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Descriptive Statistics
# MAGIC - Create descriptive statistics for the loaded data
# MAGIC - Results are loaded into pandas frame to allow for easy visibility

# COMMAND ----------

df_airlines_summary = spark.createDataFrame(df_airlines.describe().toPandas())
df_weather_summary = spark.createDataFrame(df_weather.describe().toPandas())

df_weather_summary.createOrReplaceTempView("df_weather_summary")
df_airlines_summary.createOrReplaceTempView("df_airlines_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Load Descriptive Statistics To DB
# MAGIC - Save descriptive statistics into data lake
# MAGIC - This allows everyone access to the tables

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_airlines_summary_prod_sql; CREATE TABLE df_airlines_summary_prod_sql AS SELECT * FROM df_airlines_summary; 
# MAGIC DROP TABLE IF EXISTS df_weather_summary_prod_sql; CREATE TABLE df_weather_summary_prod_sql AS SELECT * FROM df_weather_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Review Descriptive Statistics

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from df_airlines_summary_prod_sql;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from df_weather_summary_prod_sql;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 6: Dataset Balance
# MAGIC - The dataset is not balanced

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   'Delay >= 15 min' Delay_Type, count(*) Flight_CNT
# MAGIC FROM df_airlines_prd_sql
# MAGIC WHERE DEP_DEL15 = 1
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Delay < 15 min' Delay_Type, count(*) Flight_CNT
# MAGIC FROM df_airlines_prd_sql
# MAGIC WHERE DEP_DEL15 = 0

# COMMAND ----------



# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
  A.Date date,
  A.var,
  (B.cnt / cast(A.cnt as float)) * 100 as cnt
FROM 
    (
    SELECT 
      last_day(AN.FL_DATE) date, 
      lpad(cast(cast(AN.DEP_TIME/100 as int) as string),2,'0') var, 
      count(*) cnt
    FROM df_airlines_prd_sql AN
    GROUP BY last_day(AN.FL_DATE), lpad(cast(cast(AN.DEP_TIME/100 as int) as string),2,'0')
  ) A
  LEFT JOIN
  (
  SELECT 
      last_day(BN.FL_DATE) date, 
      lpad(cast(cast(BN.DEP_TIME/100 as int) as string),2,'0') var, 
      count(*) cnt
    FROM df_airlines_prd_sql BN
    WHERE BN.DEP_DEL15 = 1
    GROUP BY last_day(BN.FL_DATE), lpad(cast(cast(BN.DEP_TIME/100 as int) as string),2,'0')    
  ) B
ON A.date = B.date AND A.var = B.var
WHERE A.var is not null
ORDER BY 1
""")

DATA_DF = df_temp.toPandas()
DATA_DF_GB = DATA_DF.groupby(by=["date"]).sum()
DATA_DF_GB.reset_index(inplace=True)

col_join = ['date']
return_df = pd.merge(DATA_DF, DATA_DF_GB, left_on=col_join, right_on=col_join)
return_df["cnt"] = (return_df["cnt_x"] / return_df["cnt_y"]) * 100

DATA_DFP = return_df.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=0.5)
fig = plt.figure(figsize=(6, 4),frameon =True, dpi=200)  
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 3},fmt='0.1f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Hour of Day',title=("% of Delays by Hour of Day"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------



# COMMAND ----------

df_temp = sqlContext.sql(""" 
SELECT 
      last_day(AN.FL_DATE) date, 
      AN.OP_CARRIER var, 
      count(*)/10000 cnt
FROM df_airlines_prd_sql AN
GROUP BY last_day(AN.FL_DATE), AN.OP_CARRIER
ORDER BY 1
""")

DATA_DF = df_temp.toPandas()
DATA_DFP = DATA_DF.pivot(index='var', columns='date', values='cnt')
sns.set(font_scale=.4)
fig = plt.figure(figsize=(7, 3),frameon =True, dpi=200)     
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 3},fmt='0.0f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Operator',title=("Flights Per Month by Operator (In 10k)"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC       last_day(AN.FL_DATE) date, 
# MAGIC       count(*) cnt
# MAGIC FROM df_airlines_prd_sql AN
# MAGIC GROUP BY last_day(AN.FL_DATE)
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md ## Step 7: Principal Component Analysis (PCA)
# MAGIC 
# MAGIC PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. 
# MAGIC 
# MAGIC We have identified the weather features as an ideal feature set to use with PCA due to the highly correlated of the many measurements that make up each observation. 
# MAGIC 
# MAGIC In EDA we test out what is an ideal Components Number using the Explained Variance graph.

# COMMAND ----------

#SET CONTEXT
sc = spark.sparkContext
sqlContext = SQLContext(sc)

#GET DATA
df = sqlContext.sql("""SELECT * from airlines_with_features_sql""")

# COMMAND ----------

#SELECT FEATURES WEATHER 
numericColsWH =   ['Src_WND_0', 'Src_WND_1', 'Src_WND_3', 'Src_WND_4', 'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1', 'Src_VIS_2', 'Src_VIS_3', 'Src_TMP_0', 'Src_TMP_1', 'Src_DEW_0', 'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 'Dst_WND_0', 'Dst_WND_1', 'Dst_WND_3', 'Dst_WND_4', 'Dst_CIG_0', 'Dst_CIG_1', 'Dst_VIS_0', 'Dst_VIS_1', 'Dst_VIS_2', 'Dst_VIS_3', 'Dst_TMP_0', 'Dst_TMP_1', 'Dst_DEW_0', 'Dst_DEW_1', 'Dst_SLP_0', 'Dst_SLP_1', 'Dst_GA1_0', 'Dst_GA1_1', 'Dst_GA1_2', 'Dst_GA1_3', 'Dst_GA1_4', 'Dst_GA1_5']

# COMMAND ----------

#VECTOR THE INPUT DATA
assembler = VectorAssembler(inputCols=numericColsWH, outputCol="NumfeaturesPCA")
df = assembler.transform(df)

Scaler = MinMaxScaler(inputCol="NumfeaturesPCA", outputCol="PCA_IN_Numfeatures")
df = Scaler.fit(df).transform(df)

#RUN PCA
pca = PCA(k=15 , inputCol="PCA_IN_Numfeatures" , outputCol="PCA_OUT_Numfeatures")
model = pca.fit(df)
df = model.transform(df)

#GRAPH EXPLAINED VARIANCE
data = model.explainedVariance.values
plt.bar(range(1,16,1), data)
plt.xticks(np.arange(1, 16, 1))
plt.title("Principal Component Explained Variance VS Each Principal Component")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")

# COMMAND ----------

# MAGIC %md # Weather Data Statistics

# COMMAND ----------

# spark configuration
sc = spark.sparkContext
sqlContext = SQLContext(sc)

# COMMAND ----------

df_weather_processed = sqlContext.sql('''select * from df_weather_processed_sql''')

# COMMAND ----------

df_weather_processed_summary = spark.createDataFrame(df_weather_processed.describe().toPandas())
df_weather_processed_summary.createOrReplaceTempView("df_weather_processed_summary")

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_processed_summary_sql; CREATE TABLE df_weather_processed_summary_sql AS SELECT * FROM df_weather_processed_summary; 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_weather_processed_summary_sql
