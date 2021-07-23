# Databricks notebook source
# MAGIC %md 
# MAGIC # Final Project EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Import Packages

# COMMAND ----------

from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pyspark.mllib.stat import Statistics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import udf
import numpy as np

# COMMAND ----------

def get_dtype(df,coltype):
  col_list = []
  for name,dtype in df.dtypes:
    if dtype == coltype:
      col_list.append(name)
  return col_list

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Data
# MAGIC - Sample data is loaded into tables

# COMMAND ----------

df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

df_airlines.createOrReplaceTempView("df_airlines")
df_weather.createOrReplaceTempView("df_weather")
df_stations.createOrReplaceTempView("df_stations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Data to Data Lake
# MAGIC - Loading data into the data lake
# MAGIC - The top benefits of using the data lake: prevent Data Corruption, Faster Queries, Increase Data Freshness, Reproduce ML Models, Achieve Compliance and ability to have ACID-compliant reads and writes.

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_stations_sql; CREATE TABLE df_stations_sql AS SELECT * FROM df_stations; 
# MAGIC DROP TABLE IF EXISTS df_airlines_sql; CREATE TABLE df_airlines_sql AS SELECT * FROM df_airlines; 
# MAGIC DROP TABLE IF EXISTS df_weather_sql; CREATE TABLE df_weather_sql AS SELECT * FROM df_weather;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Descriptive Statistics
# MAGIC - Create descriptive statistics for the loaded data
# MAGIC - Results are loaded into pandas frame to allow for easy visibility

# COMMAND ----------

df_airlines_summary = spark.createDataFrame(df_airlines.describe().toPandas())
df_airlines_summary.createOrReplaceTempView("df_airlines_summary")

df_weather_summary = spark.createDataFrame(df_weather.describe().toPandas())
df_weather_summary.createOrReplaceTempView("df_weather_summary")

df_stations_summary = spark.createDataFrame(df_stations.describe().toPandas())
df_stations_summary.createOrReplaceTempView("df_stations_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Load Descriptive Statistics To Data Lake
# MAGIC - Save descriptive statistics into data lake
# MAGIC - This allows everyone access to the tables

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_airlines_summary_sql; CREATE TABLE df_airlines_summary_sql AS SELECT * FROM df_airlines_summary; 
# MAGIC DROP TABLE IF EXISTS df_weather_summary_sql; CREATE TABLE df_weather_summary_sql AS SELECT * FROM df_weather_summary;
# MAGIC DROP TABLE IF EXISTS df_stations_summary_sql; CREATE TABLE df_stations_summary_sql AS SELECT * FROM df_stations_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Review Descriptive Statistics
# MAGIC - Results for descriptive statistics
# MAGIC ### table: airlines
# MAGIC - We are working with a sample of the data
# MAGIC - Row Count = 161057
# MAGIC - Large number of features are null. Example: DIV4_WHEELS_ON
# MAGIC - We have a mixture of feature types
# MAGIC - We have panel data
# MAGIC - Will need to address cancellations as they might require special treatment
# MAGIC ### table: weather
# MAGIC - We are working with a sample of the data
# MAGIC - Row Count = 29643209
# MAGIC - Some columns will require special treatment: split by comma
# MAGIC - Fields contain special characters (+/-, N etc..)
# MAGIC - We have panel data
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
# MAGIC ### table: airlines
# MAGIC - We are working with a sample of the data
# MAGIC ### table: weather
# MAGIC - We are working with a sample of the data
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
# MAGIC ### Step 7: DataSet Balance
# MAGIC - The dataset is not balanced
# MAGIC - Non delayed flights are ~ 3 times bigger then delayed flights, potention problem as classification algorithm might simply always predict the majority class

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   'Delay >= 15 min' Delay_Type, count(*) Flight_CNT
# MAGIC FROM df_airlines_sql
# MAGIC WHERE DEP_DELAY >= 15
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Delay < 15 min' Delay_Type, count(*) Flight_CNT
# MAGIC FROM df_airlines_sql
# MAGIC WHERE DEP_DELAY < 15

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8: Delay Length
# MAGIC - Majority of the delays are under 30 minutes
# MAGIC - However some extreeme delays due exists where the delay is more than 1000 minutes, histogram has a very long tail
# MAGIC - Small group of flights leaves early

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   DEP_DELAY
# MAGIC FROM df_airlines_sql

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9: Delays by Destination
# MAGIC - Delays by destination overall seem random over time

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
    WHERE BN.DEP_DELAY >= 15
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
fig = plt.figure(figsize=(5, 40),frameon =True)   
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.2f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Destination Airport',title=("Delays by Destination"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10: Delays By Departure Time
# MAGIC - Delays appear to happen later in the day, most likely as the day progresses the chances are higher that your airplance was delayed on another trip
# MAGIC - The trend is consistant across the three sample months

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
    WHERE BN.DEP_DELAY >= 15
    GROUP BY last_day(BN.FL_DATE), lpad(cast(cast(BN.DEP_TIME/100 as int) as string),2,'0')    
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
fig = plt.figure(figsize=(10, 10),frameon =True)   
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.2f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Hour of Day',title=("% of Delays by Hour of Day"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10: Delays By Operator
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
sns.set(font_scale=1)
fig = plt.figure(figsize=(10, 10),frameon =True)   
fig = sns.heatmap(DATA_DFP,annot=True,cbar=False,linewidths=0.1,annot_kws={"size": 12},fmt='0.2f', square=False,cmap="Blues") 
fig.set(xlabel=' ', ylabel='Destination Airport',title=("% of Delays by Operator"))
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 11: Top Destination
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
# MAGIC ### Step 12: Top Origin
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
# MAGIC ## Step 13: Correlation Airlines
# MAGIC - some features are correlated and will need to be removed

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
# MAGIC ## Step 14: Weather Values Extract

# COMMAND ----------

# MAGIC %sql SELECT * FROM df_weather_sql LIMIT 2

# COMMAND ----------

# MAGIC %sql SELECT * FROM df_weather_clean LIMIT 20

# COMMAND ----------

# MAGIC %sql SELECT distinct KA1 FROM df_weather_sql LIMIT 20

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_clean_small;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_small AS
# MAGIC select * from df_weather_sql TABLESAMPLE (0.1 PERCENT);

# COMMAND ----------

# MAGIC %sql select  *   from df_weather_clean_small;
# MAGIC SHOW COLUMNS IN df_weather_clean_small;

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM df_weather_clean_piotr where length(AW1) > 0

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_clean_piotr;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_piotr AS
# MAGIC SELECT 
# MAGIC    STATION
# MAGIC   ,DATE
# MAGIC   ,YEAR(DATE) as D_YEAR
# MAGIC   ,MONTH(DATE) as D_MONTH
# MAGIC   ,DAY(DATE) as D_DAY
# MAGIC   ,HOUR(DATE) AS D_HOUR
# MAGIC   ,MINUTE(DATE) AS D_MINUTE
# MAGIC   
# MAGIC   ,SOURCE
# MAGIC   ,cast(LATITUDE as float) as LATITUDE
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
# MAGIC   --WND
# MAGIC   ,WND
# MAGIC   ,cast(SPLIT(WND,',')[0] as int) as WND_0
# MAGIC   ,cast(SPLIT(WND,',')[1] as int) as WND_1
# MAGIC   ,SPLIT(WND,',')[2] as WND_2
# MAGIC   ,cast(SPLIT(WND,',')[3] as int) as WND_3
# MAGIC   ,cast(SPLIT(WND,',')[4] as int) as WND_4
# MAGIC   
# MAGIC   --CIG
# MAGIC   ,CIG
# MAGIC   ,cast(SPLIT(CIG,',')[0] as int) as CIG_0
# MAGIC   ,cast(SPLIT(CIG,',')[1] as int) as CIG_1
# MAGIC   ,SPLIT(CIG,',')[2] as CIG_2
# MAGIC   ,SPLIT(CIG,',')[3] as CIG_3
# MAGIC   
# MAGIC   --VIS
# MAGIC   ,VIS
# MAGIC   ,cast(SPLIT(VIS,',')[0] as int) as VIS_0
# MAGIC   ,cast(SPLIT(VIS,',')[1] as int) as VIS_1
# MAGIC   ,cast(SPLIT(VIS,',')[2] as int) as VIS_2
# MAGIC   ,cast(SPLIT(VIS,',')[3] as int) as VIS_3
# MAGIC   
# MAGIC   --TMP
# MAGIC   ,TMP
# MAGIC   ,cast(substring(SPLIT(TMP,',')[0],2) as int)  as TMP_0
# MAGIC   ,substring(TMP,0,1)                           as TMP_0A
# MAGIC   ,cast( SPLIT(TMP,',')[1] as int)              as TMP_1
# MAGIC   
# MAGIC   --DEW
# MAGIC   ,DEW
# MAGIC   ,cast(substring(SPLIT(DEW,',')[0],2) as int)  as DEW_0
# MAGIC   ,substring(DEW,0,1)                           as DEW_0A
# MAGIC   ,cast( SPLIT(DEW,',')[1] as int)              as DEW_1
# MAGIC 
# MAGIC   --SLP
# MAGIC   ,SLP
# MAGIC   ,cast( SPLIT(SLP,',')[0] as int)              as SLP_0
# MAGIC   ,cast( SPLIT(SLP,',')[1] as int)              as SLP_1
# MAGIC 
# MAGIC   --AW1
# MAGIC   ,AW1
# MAGIC   
# MAGIC   --GA1
# MAGIC   ,GA1
# MAGIC   ,cast( SPLIT(GA1,',')[0] as int)              as GA1_0
# MAGIC   ,cast( SPLIT(GA1,',')[1] as int)              as GA1_1  
# MAGIC   ,cast(substring(SPLIT(GA1,',')[2],2) as int)  as GA1_2
# MAGIC   ,substring(SPLIT(GA1,',')[2],0,1)             as GA1_2A
# MAGIC   ,cast( SPLIT(GA1,',')[3] as int)              as GA1_3  
# MAGIC   ,cast( SPLIT(GA1,',')[4] as int)              as GA1_4  
# MAGIC   ,cast( SPLIT(GA1,',')[5] as int)              as GA1_5  
# MAGIC     
# MAGIC   --GA2
# MAGIC   ,GA2
# MAGIC   ,cast( SPLIT(GA2,',')[0] as int)              as GA2_0
# MAGIC   ,cast( SPLIT(GA2,',')[1] as int)              as GA2_1  
# MAGIC   ,cast(substring(SPLIT(GA2,',')[2],2) as int)  as GA2_2
# MAGIC   ,substring(SPLIT(GA2,',')[2],0,1)             as GA2_2A
# MAGIC   ,cast( SPLIT(GA2,',')[3] as int)              as GA2_3  
# MAGIC   ,cast( SPLIT(GA2,',')[4] as int)              as GA2_4  
# MAGIC   ,cast( SPLIT(GA2,',')[5] as int)              as GA2_5  
# MAGIC   
# MAGIC   --GA3
# MAGIC   ,GA3
# MAGIC   ,cast( SPLIT(GA3,',')[0] as int)              as GA3_0
# MAGIC   ,cast( SPLIT(GA3,',')[1] as int)              as GA3_1  
# MAGIC   ,cast(substring(SPLIT(GA3,',')[2],2) as int)  as GA3_2
# MAGIC   ,substring(SPLIT(GA3,',')[2],0,1)             as GA3_2A
# MAGIC   ,cast( SPLIT(GA3,',')[3] as int)              as GA3_3  
# MAGIC   ,cast( SPLIT(GA3,',')[4] as int)              as GA3_4  
# MAGIC   ,cast( SPLIT(GA3,',')[5] as int)              as GA3_5  
# MAGIC   
# MAGIC   --GA4
# MAGIC   ,GA4
# MAGIC   ,cast( SPLIT(GA4,',')[0] as int)              as GA4_0
# MAGIC   ,cast( SPLIT(GA4,',')[1] as int)              as GA4_1  
# MAGIC   ,cast(substring(SPLIT(GA4,',')[2],2) as int)  as GA4_2
# MAGIC   ,substring(SPLIT(GA4,',')[2],0,1)             as GA4_2A
# MAGIC   ,cast( SPLIT(GA4,',')[3] as int)              as GA4_3  
# MAGIC   ,cast( SPLIT(GA4,',')[4] as int)              as GA4_4  
# MAGIC   ,cast( SPLIT(GA4,',')[5] as int)              as GA4_5  
# MAGIC   
# MAGIC   --GE1  
# MAGIC   ,GE1
# MAGIC   ,cast( SPLIT(GE1,',')[0] as int)              as GE1_0
# MAGIC   ,SPLIT(GE1,',')[1]                            as GE1_1
# MAGIC   ,cast( SPLIT(GE1,',')[2] as int)              as GE1_2
# MAGIC   ,substring(SPLIT(GE1,',')[2],0,1)             as GE1_2A
# MAGIC   ,cast( SPLIT(GE1,',')[3] as int)              as GE1_3
# MAGIC   ,substring(SPLIT(GE1,',')[3],0,1)             as GE1_3A
# MAGIC   
# MAGIC   --GF1
# MAGIC    ,GF1
# MAGIC    ,cast( SPLIT(GF1,',')[0] as int)              as GF1_0
# MAGIC    ,cast( SPLIT(GF1,',')[1] as int)              as GF1_1
# MAGIC    ,cast( SPLIT(GF1,',')[2] as int)              as GF1_2
# MAGIC    ,cast( SPLIT(GF1,',')[3] as int)              as GF1_3
# MAGIC    ,cast( SPLIT(GF1,',')[4] as int)              as GF1_4
# MAGIC    ,cast( SPLIT(GF1,',')[5] as int)              as GF1_5
# MAGIC    ,cast( SPLIT(GF1,',')[6] as int)              as GF1_6
# MAGIC    ,cast( SPLIT(GF1,',')[7] as int)              as GF1_7
# MAGIC    ,cast( SPLIT(GF1,',')[8] as int)              as GF1_8
# MAGIC    ,cast( SPLIT(GF1,',')[9] as int)              as GF1_9
# MAGIC    ,cast( SPLIT(GF1,',')[10] as int)              as GF1_10
# MAGIC    ,cast( SPLIT(GF1,',')[11] as int)              as GF1_11
# MAGIC    ,cast( SPLIT(GF1,',')[12] as int)              as GF1_12
# MAGIC    
# MAGIC    --KA1
# MAGIC    ,KA1
# MAGIC    ,cast( SPLIT(KA1,',')[0] as int)              as KA1_0
# MAGIC    ,SPLIT(KA1,',')[1]                            as KA1_1
# MAGIC    ,cast( SPLIT(KA1,',')[2] as int)              as KA1_2
# MAGIC    ,substring(SPLIT(KA1,',')[2],0,1)             as KA1_2A
# MAGIC    ,cast( SPLIT(KA1,',')[3] as int)              as KA1_3
# MAGIC    
# MAGIC   
# MAGIC FROM df_weather_sql --df_weather_clean_small

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_stg_sql;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_piotr AS
# MAGIC SELECT 
# MAGIC    STATION
# MAGIC   ,DATE
# MAGIC   ,YEAR(DATE) as D_YEAR
# MAGIC   ,MONTH(DATE) as D_MONTH
# MAGIC   ,DAY(DATE) as D_DAY
# MAGIC   ,HOUR(DATE) AS D_HOUR
# MAGIC   ,MINUTE(DATE) AS D_MINUTE
# MAGIC   
# MAGIC   ,SOURCE
# MAGIC   ,cast(LATITUDE as float) as LATITUDE
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
# MAGIC   --WND
# MAGIC   ,WND
# MAGIC   ,cast(SPLIT(WND,',')[0] as int) as WND_0
# MAGIC   ,cast(SPLIT(WND,',')[1] as int) as WND_1
# MAGIC   ,SPLIT(WND,',')[2] as WND_2
# MAGIC   ,cast(SPLIT(WND,',')[3] as int) as WND_3
# MAGIC   ,cast(SPLIT(WND,',')[4] as int) as WND_4
# MAGIC   
# MAGIC   --CIG
# MAGIC   ,CIG
# MAGIC   ,cast(SPLIT(CIG,',')[0] as int) as CIG_0
# MAGIC   ,cast(SPLIT(CIG,',')[1] as int) as CIG_1
# MAGIC   ,SPLIT(CIG,',')[2] as CIG_2
# MAGIC   ,SPLIT(CIG,',')[3] as CIG_3
# MAGIC   
# MAGIC   --VIS
# MAGIC   ,VIS
# MAGIC   ,cast(SPLIT(VIS,',')[0] as int) as VIS_0
# MAGIC   ,cast(SPLIT(VIS,',')[1] as int) as VIS_1
# MAGIC   ,cast(SPLIT(VIS,',')[2] as int) as VIS_2
# MAGIC   ,cast(SPLIT(VIS,',')[3] as int) as VIS_3
# MAGIC   
# MAGIC   --TMP
# MAGIC   ,TMP
# MAGIC   ,cast(substring(SPLIT(TMP,',')[0],2) as int)  as TMP_0
# MAGIC   ,substring(TMP,0,1)                           as TMP_0A
# MAGIC   ,cast( SPLIT(TMP,',')[1] as int)              as TMP_1
# MAGIC   
# MAGIC   --DEW
# MAGIC   ,DEW
# MAGIC   ,cast(substring(SPLIT(DEW,',')[0],2) as int)  as DEW_0
# MAGIC   ,substring(DEW,0,1)                           as DEW_0A
# MAGIC   ,cast( SPLIT(DEW,',')[1] as int)              as DEW_1
# MAGIC 
# MAGIC   --SLP
# MAGIC   ,SLP
# MAGIC   ,cast( SPLIT(SLP,',')[0] as int)              as SLP_0
# MAGIC   ,cast( SPLIT(SLP,',')[1] as int)              as SLP_1
# MAGIC 
# MAGIC   --AW1
# MAGIC   ,AW1
# MAGIC   
# MAGIC   --GA1
# MAGIC   ,GA1
# MAGIC   ,cast( SPLIT(GA1,',')[0] as int)              as GA1_0
# MAGIC   ,cast( SPLIT(GA1,',')[1] as int)              as GA1_1  
# MAGIC   ,cast(substring(SPLIT(GA1,',')[2],2) as int)  as GA1_2
# MAGIC   ,substring(SPLIT(GA1,',')[2],0,1)             as GA1_2A
# MAGIC   ,cast( SPLIT(GA1,',')[3] as int)              as GA1_3  
# MAGIC   ,cast( SPLIT(GA1,',')[4] as int)              as GA1_4  
# MAGIC   ,cast( SPLIT(GA1,',')[5] as int)              as GA1_5  
# MAGIC     
# MAGIC   --GA2
# MAGIC   ,GA2
# MAGIC   ,cast( SPLIT(GA2,',')[0] as int)              as GA2_0
# MAGIC   ,cast( SPLIT(GA2,',')[1] as int)              as GA2_1  
# MAGIC   ,cast(substring(SPLIT(GA2,',')[2],2) as int)  as GA2_2
# MAGIC   ,substring(SPLIT(GA2,',')[2],0,1)             as GA2_2A
# MAGIC   ,cast( SPLIT(GA2,',')[3] as int)              as GA2_3  
# MAGIC   ,cast( SPLIT(GA2,',')[4] as int)              as GA2_4  
# MAGIC   ,cast( SPLIT(GA2,',')[5] as int)              as GA2_5  
# MAGIC   
# MAGIC   --GA3
# MAGIC   ,GA3
# MAGIC   ,cast( SPLIT(GA3,',')[0] as int)              as GA3_0
# MAGIC   ,cast( SPLIT(GA3,',')[1] as int)              as GA3_1  
# MAGIC   ,cast(substring(SPLIT(GA3,',')[2],2) as int)  as GA3_2
# MAGIC   ,substring(SPLIT(GA3,',')[2],0,1)             as GA3_2A
# MAGIC   ,cast( SPLIT(GA3,',')[3] as int)              as GA3_3  
# MAGIC   ,cast( SPLIT(GA3,',')[4] as int)              as GA3_4  
# MAGIC   ,cast( SPLIT(GA3,',')[5] as int)              as GA3_5  
# MAGIC   
# MAGIC   --GA4
# MAGIC   ,GA4
# MAGIC   ,cast( SPLIT(GA4,',')[0] as int)              as GA4_0
# MAGIC   ,cast( SPLIT(GA4,',')[1] as int)              as GA4_1  
# MAGIC   ,cast(substring(SPLIT(GA4,',')[2],2) as int)  as GA4_2
# MAGIC   ,substring(SPLIT(GA4,',')[2],0,1)             as GA4_2A
# MAGIC   ,cast( SPLIT(GA4,',')[3] as int)              as GA4_3  
# MAGIC   ,cast( SPLIT(GA4,',')[4] as int)              as GA4_4  
# MAGIC   ,cast( SPLIT(GA4,',')[5] as int)              as GA4_5  
# MAGIC   
# MAGIC   --GE1  
# MAGIC   ,GE1
# MAGIC   ,cast( SPLIT(GE1,',')[0] as int)              as GE1_0
# MAGIC   ,SPLIT(GE1,',')[1]                            as GE1_1
# MAGIC   ,cast( SPLIT(GE1,',')[2] as int)              as GE1_2
# MAGIC   ,substring(SPLIT(GE1,',')[2],0,1)             as GE1_2A
# MAGIC   ,cast( SPLIT(GE1,',')[3] as int)              as GE1_3
# MAGIC   ,substring(SPLIT(GE1,',')[3],0,1)             as GE1_3A
# MAGIC   
# MAGIC   --GF1
# MAGIC    ,GF1
# MAGIC    ,cast( SPLIT(GF1,',')[0] as int)              as GF1_0
# MAGIC    ,cast( SPLIT(GF1,',')[1] as int)              as GF1_1
# MAGIC    ,cast( SPLIT(GF1,',')[2] as int)              as GF1_2
# MAGIC    ,cast( SPLIT(GF1,',')[3] as int)              as GF1_3
# MAGIC    ,cast( SPLIT(GF1,',')[4] as int)              as GF1_4
# MAGIC    ,cast( SPLIT(GF1,',')[5] as int)              as GF1_5
# MAGIC    ,cast( SPLIT(GF1,',')[6] as int)              as GF1_6
# MAGIC    ,cast( SPLIT(GF1,',')[7] as int)              as GF1_7
# MAGIC    ,cast( SPLIT(GF1,',')[8] as int)              as GF1_8
# MAGIC    ,cast( SPLIT(GF1,',')[9] as int)              as GF1_9
# MAGIC    ,cast( SPLIT(GF1,',')[10] as int)              as GF1_10
# MAGIC    ,cast( SPLIT(GF1,',')[11] as int)              as GF1_11
# MAGIC    ,cast( SPLIT(GF1,',')[12] as int)              as GF1_12
# MAGIC    
# MAGIC    --KA1
# MAGIC    ,KA1
# MAGIC    ,cast( SPLIT(KA1,',')[0] as int)              as KA1_0
# MAGIC    ,SPLIT(KA1,',')[1]                            as KA1_1
# MAGIC    ,cast( SPLIT(KA1,',')[2] as int)              as KA1_2
# MAGIC    ,substring(SPLIT(KA1,',')[2],0,1)             as KA1_2A
# MAGIC    ,cast( SPLIT(KA1,',')[3] as int)              as KA1_3 
# MAGIC FROM df_weather

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW COLUMNS IN df_weather_sql;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 14: Weather Correlation
# MAGIC - weather table features have high correlation and some will need to be removed

# COMMAND ----------

df_weather_clean = sqlContext.sql(""" SELECT * FROM df_weather_clean""")
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
# MAGIC ### Step 15: Delays by Day Of Week
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
# MAGIC ### Step 16: Delays by Trip Group
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
# MAGIC ### Step 17: Join Tables - Correlation TODO

# COMMAND ----------

# MAGIC %md # ================ MASTER JOIN =======================

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM main_flights_data limit 2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_weather_sql limit 2

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS main_flights_data_small;
# MAGIC 
# MAGIC CREATE TABLE main_flights_data_small AS
# MAGIC SELECT
# MAGIC to_timestamp(concat(CAST(YEAR AS STRING),'-',CAST(MONTH AS STRING),'-',CAST(DAY_OF_MONTH AS STRING),'T',HOUR(dep_datetime_scheduled_utc),':',floor(minute(dep_datetime_scheduled_utc)/10)*10,':00.000+0000')) DATE_JOIN
# MAGIC ,
# MAGIC *
# MAGIC FROM main_flights_data
# MAGIC WHERE YEAR = 2016 AND MONTH = 7

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_clean_small;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_small AS
# MAGIC select * from df_weather_sql 
# MAGIC WHERE D_YEAR = 2016 AND D_MONTH = 7

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC DATE,
# MAGIC DATE +  INTERVAL 2 hours as HOUR_2,
# MAGIC DATE +  INTERVAL -2 hours as HOUR_3,
# MAGIC DATE +  INTERVAL -2 minutes as HOUR_3,
# MAGIC floor(minute(DATE)/10)*10 Shave_Minutes2,
# MAGIC concat(CAST(D_YEAR AS STRING),'-',CAST(D_MONTH AS STRING),'-',CAST(D_DAY AS STRING),'T',HOUR(DATE),':',floor(minute(DATE)/10)*10,':00.000+0000') D1,
# MAGIC to_timestamp(concat(CAST(D_YEAR AS STRING),'-',CAST(D_MONTH AS STRING),'-',CAST(D_DAY AS STRING),'T',HOUR(DATE),':',floor(minute(DATE)/10)*10,':00.000+0000')) DM,
# MAGIC * 
# MAGIC from df_weather_clean_small

# COMMAND ----------

# MAGIC %md # UPDATE

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_clean_small2;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_small2 AS
# MAGIC select 
# MAGIC to_timestamp(concat(CAST(D_YEAR AS STRING),'-',CAST(D_MONTH AS STRING),'-',CAST(D_DAY AS STRING),'T',HOUR(DATE),':',floor(minute(DATE)/10)*10,':00.000+0000')) AS DATE_JOIN,
# MAGIC * 
# MAGIC from df_weather_clean_small

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_clean_small3;
# MAGIC 
# MAGIC CREATE TABLE df_weather_clean_small3 AS
# MAGIC select 
# MAGIC DATE_JOIN +  INTERVAL -2 hours as DATE_JOIN_2H,
# MAGIC * 
# MAGIC from df_weather_clean_small2

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from main_flights_data_small AL1
# MAGIC LEFT JOIN df_weather_clean_small3 AL2 
# MAGIC   ON AL1.DATE_JOIN = AL2.DATE_JOIN_2H AND AL1.station_origin = AL2.STATION
# MAGIC ORDER BY AL1.station_origin, AL1.DATE_JOIN

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from main_flights_data_small AL1
# MAGIC WHERE DATE_JOIN = '2016-07-28T16:30:00.000+0000' and station_origin = 70133026616

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_weather_clean_small3 AL1
# MAGIC WHERE STATION = 70133026616
# MAGIC ORDER BY DATE_JOIN_2H

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_weather_sql
# MAGIC WHERE STATION = 70133026616
# MAGIC ORDER BY DATE

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from main_flights_data AL1
# MAGIC LEFT JOIN df_weather_sql AL2 
# MAGIC   ON date_trunc("hour",AL1.dep_datetime_scheduled_utc) = date_trunc("hour",AL2.DATE)
# MAGIC   and AL1.station_origin = AL2.STATION
# MAGIC WHERE AL2.Station is null
# MAGIC union all
# MAGIC select count(*) from main_flights_data AL1
# MAGIC LEFT JOIN df_weather_sql AL2 
# MAGIC   ON date_trunc("hour",AL1.dep_datetime_scheduled_utc) = date_trunc("hour",AL2.DATE)
# MAGIC   and AL1.station_origin = AL2.STATION
# MAGIC WHERE AL2.Station is not null
# MAGIC union all
# MAGIC select count(*) from main_flights_data

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC date_trunc("year",dep_datetime_scheduled_utc), 
# MAGIC date_trunc("month",dep_datetime_scheduled_utc), 
# MAGIC date_trunc("day",dep_datetime_scheduled_utc), 
# MAGIC date_trunc("hour",dep_datetime_scheduled_utc), 
# MAGIC date_trunc("minute",dep_datetime_scheduled_utc), 
# MAGIC date_trunc("minute",dep_datetime_scheduled_utc), 
# MAGIC * from main_flights_data_small

# COMMAND ----------

# MAGIC %md # Weather ETL

# COMMAND ----------

!pip install timezonefinder

# COMMAND ----------

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType

import numpy as np
import math
from timezonefinder import TimezoneFinder

# COMMAND ----------

# load weather dataset
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")

# keep only the data coming from the stations in the US
df_weather = df_weather.where(col('NAME').endswith('US'))

# print number of records
print(f'Number of weather records loaded: {df_weather.count()}')

# select columns to keep
weather_columns = ['STATION', 'DATE', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND', \
                   'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AW1', 'GA1', 'GA2', 'GA3', 'GA4', 'GE1', 'GF1', 'KA1']

# keep relevant columns
# df_weather = df_weather.select(*weather_columns)

# print number of columns
print(f'Number of columns in weather dataset: {len(df_weather.columns)}')

# unpack columns containing comma separated values
df_weather.createOrReplaceTempView("df_weather")

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_sql;
# MAGIC 
# MAGIC CREATE TABLE df_weather_sql AS
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
# MAGIC FROM df_weather
# MAGIC WHERE STATION IS NOT NULL

# COMMAND ----------

print(get_dtype(df_weather_sql,"int"))

# COMMAND ----------

def get_dtype(df,coltype):
  col_list = []
  for name,dtype in df.dtypes:
    if dtype == coltype:
      col_list.append(name)
  return col_list

df_weather_sql = sqlContext.sql("""SELECT * FROM df_weather_sql""")

sum_var = ['WND_0', 'WND_1', 'WND_3', 'WND_4', 'CIG_0', 'CIG_1', 'VIS_0', 'VIS_1', 'VIS_2', 'VIS_3', 'TMP_0', 'TMP_1', 'DEW_0', 'DEW_1', 'SLP_0', 'SLP_1', 'GA1_0', 'GA1_1', 'GA1_2', 'GA1_3', 'GA1_4', 'GA1_5']
group_col = ['DATE_JOIN_MN','DATE_JOIN_HH','DATE_JOIN_DA']
group_col = ['DATE_JOIN_MN']
first_run_flag = 0

for g, gval in enumerate(group_col):
  for i, cval in enumerate(sum_var):
    query = """
    select 
       STATION
      ,"""+gval+""" +  INTERVAL -2 hours as DATE_JOIN_2H
      ,avg(""" +cval+""") as feature_name
      ,count(*) OBS_CNT
      ,'""" + cval + """' as Col_Name
      ,'""" + gval + """' as Sum_Name
    FROM df_weather_sql
    WHERE """ + cval + """ NOT IN (999,9999,99999,999999)
    GROUP BY STATION 
            ,"""+gval+""" +  INTERVAL -2 hours
    """
    result = sqlContext.sql(query)
    if first_run_flag == 0:
      final_result = result
      first_run_flag = 1
    else:    
      final_result = final_result.union(result)
final_result.createOrReplaceTempView("final_result")
display(final_result)

# COMMAND ----------

spark.conf.set("spark.sql.optimizer.maxIterations", "1000")
spark.conf.set("spark.sql.analyzer.maxIterations", "1000")

SQL_JOIN = ""
SQL_TOP = "select\n A.STATION\n,A.DATE_JOIN_2H\n"
SQL_FROM = """FROM (Select distinct A_I.STATION, A_I.DATE_JOIN_2H FROM final_result A_I) A"""

for i, val in enumerate(sum_var):
  i = str(i)
  SQL_JOIN = SQL_JOIN + """ LEFT JOIN (select A1_""" +i+ """.STATION, A1_""" +i+ """.DATE_JOIN_2H, A1_""" +i+ """.feature_name FROM final_result A1_""" +i+ """ WHERE A1_""" +i+ """.Col_Name ='"""+val +"""' ) AL""" +i+ """ ON A.STATION = AL""" +i+ """.STATION AND A.DATE_JOIN_2h = AL""" +i+ """.DATE_JOIN_2H \n"""
  SQL_TOP = SQL_TOP + """,AL""" +i+ """.feature_name AS """+val+""" \n"""
      
FINAL_SQL = SQL_TOP + SQL_FROM + SQL_JOIN
df_weather_summary = sqlContext.sql(FINAL_SQL)
df_weather_summary.createOrReplaceTempView("df_weather_summary")
df_weather_summary_stats = spark.createDataFrame(df_weather_summary.describe().toPandas())

# COMMAND ----------

df_weather_summary_stats = spark.createDataFrame(df_weather_summary.describe().toPandas())

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_weather_summary_stats

# COMMAND ----------

df_weather_summary_filled = df_weather_summary.na.fill(value=5679,subset=["VIS_2"])

# COMMAND ----------



# COMMAND ----------

#data path to our directory 
path = 'dbfs:/mnt/Azure/'
display(dbutils.fs.ls(path))

# save processed weather data. Run when finish with transformations.
dbutils.fs.rm(path + "/processed_data/" + "weather_processed_detail.parquet", recurse =True)
temp.write.format("parquet").save(path + "/processed_data/" + "weather_processed_detail.parquet")
