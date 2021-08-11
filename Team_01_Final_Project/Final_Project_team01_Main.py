# Databricks notebook source
# MAGIC %md # Flight departure delay predictions
# MAGIC ## Main Notebook
# MAGIC 
# MAGIC Team members:
# MAGIC - Isabel Garcia Pietri
# MAGIC - Madhu Hegde
# MAGIC - Amit Karandikar
# MAGIC - Piotr Parkitny

# COMMAND ----------

# MAGIC %md ##  Problem Statement
# MAGIC 
# MAGIC Flight delays represent a big issue in the air travel industry. Delays cause huge losses to airlines, impact passenger satisfaction and cause complex logistic problems. Because of this, it is crucial for airlines to better understand flight delays, in order to minimize the impact in their business.
# MAGIC 
# MAGIC There are many things that can cause a flight to be delayed, going from aircraft maintenance, extreme weather, airport operations, security and screening activities, a previous flight delay causing the next flight to depart late, among many others. Because there are many moving parts in the air travel operations, solving the delay problem is quite complex.
# MAGIC 
# MAGIC Many people have worked in understanding the problem of flights delays. For instance, in 2019 Navoneel Chakrabarty (1) developed a model to predict arrival delay for American Airlines flights in US, that achieved an accuracy of 85.73%, which is considered a very good accuracy for this type of problems.
# MAGIC 
# MAGIC The implementation of a model to predict flight delays would have a great impact on airline operations. Airlines would be able to minimize the impact of such delays by making changes on passenger itineraries, flight routes, crew assignments, aircraft schedules and maintenance, etc.
# MAGIC 
# MAGIC The main purpose of this study is to create a model to predict departure delay for flights in the US, where a delay is defined as 15-minute delay or more. We use a subset of the of the flight's on-time performance data provided by the United States Bureau of Transportation Statistics. The data comprises data of flights departing from all major US airports for the 2015-2019 timeframe. Additionally, we use weather information from the National Oceanic and Atmospheric Administration repository.
# MAGIC 
# MAGIC The **output variable** in our model is a binary variable, where 1 represent flights that experienced departure delay and 0 represent flights with on-time departure.
# MAGIC 
# MAGIC #### About the performance metrics
# MAGIC 
# MAGIC Precision is the ratio between true positives and all the predicted positives. For our problem statement, is the measure of flights that we correctly identify as delayed out of all flights that are predicted as delayed. 
# MAGIC 
# MAGIC $$Precision = \frac{TP}{TP + FP}$$
# MAGIC 
# MAGIC Recall measures if our model is correctly identifying true positives. For our problem statement: of all the flights that are actually delayed, how many we correctly identified as delayed. 
# MAGIC $$Recall = \frac{TP}{TP + FN}$$
# MAGIC 
# MAGIC In order to minimize economic losses due to delays, airlines would like to avoid situations where the flight is going to be delayed, but the model classifies it not having a delay. This requires a model with high recall. On the other hand, a high precision is also important. 
# MAGIC The cases where the flight is not going to be delayed and the model predicts a delay, also represent an economic impact, since airlines will implement actions such as letting customers know so they don’t show up at the airport so early, changing passengers’ itineraries, etc. All these actions cause economic/reputation damage when implemented in cases where in reality there is no delay.
# MAGIC 
# MAGIC Because all this, we aim to balance precision and recall. Hence, in this study we use f1-score as the main metric to optimize. 
# MAGIC 
# MAGIC $$f1 = 2\times\frac{Precision \times Recall}{Precision + Recall} $$

# COMMAND ----------

# MAGIC %md ## Package imports, directories and configuration

# COMMAND ----------

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, expr, unix_timestamp, avg, round, lag, hour
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

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_colwidth', 999)
pd.set_option('display.max_columns', 999)

from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, expr, unix_timestamp, avg, round, lag, to_date, row_number, explode, array, lit
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType

import numpy as np
import math

import pandas as pd
import matplotlib.pyplot as plt
 
from numpy import savetxt
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pyspark.ml.feature import PCA

# COMMAND ----------

#data path to our directory 
path = 'dbfs:/mnt/Azure/'
display(dbutils.fs.ls(path))

# COMMAND ----------

# spark configuration
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md ## Exploratory data analysis (EDA)
# MAGIC 
# MAGIC The EDA of the project is available in the notebook: https://adb-6759024569771990.10.azuredatabricks.net/?o=6759024569771990#notebook/3743280196533040/command/3743280196533041

# COMMAND ----------

# MAGIC %md ## Extract, transform, load (ETL)
# MAGIC 
# MAGIC The ETL of the project is available in the notebook: https://adb-6759024569771990.10.azuredatabricks.net/?o=6759024569771990#notebook/1407650788079970/command/1407650788079986
# MAGIC 
# MAGIC This notebook contains the ETL pipeline we use to pre-process airlines and weather data.

# COMMAND ----------

# MAGIC %md ## Feature engineering
# MAGIC 
# MAGIC Based on the results of the EDA, we designed some features around the aspects that appear to affect the most the departure delay.
# MAGIC 
# MAGIC - Operating carrier appears to be important. So, we created a feature that represents the carrier delay over a time range before prediction time.
# MAGIC - We also created features to measure an airport status: airport delay over a time range before prediction time, number of flights scheduled, among others.
# MAGIC - Additionally, we created a feature to follow the delay of an aircraft. If previous flights have been delayed, then chances are that the next flight is going to be delayed.
# MAGIC - To measure the airport’s importance, we calculated the PageRank statistic. Airports with more connections will be ranked higher than airports with few connections.
# MAGIC - We have more than 350 airports in the dataset. So, instead of using all the categories for the origin and the destination airport, we decided to use a variable that represents the size of the airport. However, for the categorical variable that represent the carriers, we kept all the categories, because we think that different carriers display different performance and we want to be able to model that. We have a total of 19 operating carriers in the dataset.
# MAGIC - In this section we also handle null values for the variables we consider for the modeling part.

# COMMAND ----------

# Load airlines pre-processed data. 
df_airlines = spark.read.parquet("/mnt/Azure/processed_data/airlines_processed.parquet")

print(f'{df_airlines.count():} flight records loaded')

# Load weather pre-processed data. 
df_weather = spark.read.parquet("/mnt/Azure/processed_data/weather_processed.parquet")

print(f'{df_weather.count():} weather records loaded')

# COMMAND ----------

# MAGIC %md ### Carrier delay
# MAGIC 
# MAGIC This feature represents the average delay a carrier has over a 12-hours period before prediction time.
# MAGIC 
# MAGIC The 12-hours window goes from 2 hours and 15 minutes to 14 hours and 15 minutes before the scheduled departure time. The reason why we consider 2 hours and 15 minutes instead of 2 hours is:
# MAGIC 
# MAGIC Let's say that we are getting statistics for a flight that is schedule to depart at 3 pm. If we use a 2-hour offset, then we would include information of flights that were scheduled to depart at 1 pm and before. But if a flight is scheduled to depart at exactly 1 pm, we would not know if that flight is delayed until 1:15 pm, so it would be wrong to use the information of this flight. To solve this, we get information of flights that are scheduled to depart 2 hours and 15 minutes before.

# COMMAND ----------

# carrier delay 12 hours window before prediction time
# carrier delay 14 hours and 15 minutes (-51,300 seconds) to 2 hours and 15 minutes (-8,100 seconds) before scheduled departure time

carrier_window = Window.partitionBy('OP_CARRIER')\
                       .orderBy(unix_timestamp('dep_datetime_scheduled_utc'))\
                       .rangeBetween(-51300, -8100)

df_airlines = df_airlines.withColumn('carrier_delay', round(avg(col('DEP_DEL15')).over(carrier_window),4) )

# COMMAND ----------

# MAGIC %md ### Origin airport delay
# MAGIC This feature represents the average delay the origin airport has over a 12-hours period before prediction time. We use the same logic explained above for the time range.

# COMMAND ----------

# airport delay 12 hours window before prediction time
# airport delay 14 hours and 15 minutes (-51,300 seconds) to 2 hours and 15 minutes (-8,100 seconds) before scheduled departure time
airport_window = Window.partitionBy('ORIGIN')\
                       .orderBy(unix_timestamp('dep_datetime_scheduled_utc'))\
                       .rangeBetween(-51300, -8100)

df_airlines = df_airlines.withColumn('airport_delay', round(avg(col('DEP_DEL15')).over(airport_window),4) )

# COMMAND ----------

# MAGIC %md ### Latest known aircraft departure status
# MAGIC This feature represents the last known status of an aircraft at departure (delayed/not delayed) before prediction time. Again, we use status of flights that were scheduled to departure at least 2 hours and 15 minutes before.

# COMMAND ----------

# create a column that represents the cutoff time to get previous flights information
# 2 hours and 15 minutes (-8,100 seconds) before the scheduled departure time
df_airlines = df_airlines.withColumn('cutoff_time_utc',(unix_timestamp(col('dep_datetime_scheduled_utc')) - 8100).cast('timestamp'))

# feature that represents the latest known departure status of an aircraft (delayed/not delayed) before the prediction time 

aircraft_depar_window = Window.partitionBy('TAIL_NUM')\
                              .orderBy('dep_datetime_scheduled_utc') 

# (lag window function returns the value that is offset rows before the current row)
# First, look the previous departure, check if departure time is before cutoff time
# if so, get the departure delay status 
# If the previous departure is not before the cutoff time, look for the previous flight, check departure time again, and so on .. 
# We go up to 5 flights before current flight to account for short flights
# When tail number is not available this feature is null
df_airlines = df_airlines.withColumn('dep_delay_aircraft', \
                                   when(col('TAIL_NUM').isNull(), None) \
                                  .when(unix_timestamp(lag(col('dep_datetime_scheduled_utc'),1,None).over(aircraft_depar_window))<\
                                                       unix_timestamp(col('cutoff_time_utc')), \
                                        lag(col('DEP_DEL15'),1,None).over(aircraft_depar_window)) 
                                  .when(unix_timestamp(lag(col('dep_datetime_scheduled_utc'),2,None).over(aircraft_depar_window))<\
                                                       unix_timestamp(col('cutoff_time_utc')), \
                                        lag(col('DEP_DEL15'),2,None).over(aircraft_depar_window) )                                                                                                   .when(unix_timestamp(lag(col('dep_datetime_scheduled_utc'),3,None).over(aircraft_depar_window))<\
                                                       unix_timestamp(col('cutoff_time_utc')), \
                                        lag(col('DEP_DEL15'),3,None).over(aircraft_depar_window) )                                                                                                   .when(unix_timestamp(lag(col('dep_datetime_scheduled_utc'),4,None).over(aircraft_depar_window))<\
                                                       unix_timestamp(col('cutoff_time_utc')), \
                                        lag(col('DEP_DEL15'),4,None).over(aircraft_depar_window) )                                                                                                   .when(unix_timestamp(lag(col('dep_datetime_scheduled_utc'),5,None).over(aircraft_depar_window))<\
                                                       unix_timestamp(col('cutoff_time_utc')), \
                                        lag(col('DEP_DEL15'),5,None).over(aircraft_depar_window) ) )

# COMMAND ----------

# MAGIC %md ### Flight number for the aircraft on a day
# MAGIC 
# MAGIC This feature represents the number of the flight for the aircraft on a day: first flight = 1, second flight= 2 and so on. 

# COMMAND ----------

# partition by tail number an date, order by datetime
aircraft_depar_window2 = Window.partitionBy('TAIL_NUM', to_date('dep_datetime_scheduled_utc'))\
                               .orderBy('dep_datetime_scheduled_utc') 

# get sequence of flights per aircraft during a day
df_airlines = df_airlines.withColumn('flight_aircraft', when(col('TAIL_NUM').isNull(), None)\
                                     .otherwise(row_number().over(aircraft_depar_window2)))

# COMMAND ----------

# MAGIC %md ### Origin and destination airport usage
# MAGIC These two features represent how busy an airport is: number of scheduled departures at origin and number of scheduled arrivals at destination in a two-hours window (1 hour before and 1 hour after the prediction time). Since we are using **scheduled** flights we can go 1 hour over the prediction time.

# COMMAND ----------

# number of flights departures scheduled between t-1 and t+1 (t scheduled departure time)
# 1 hour before: -7,200 seconds. 1 hour after: 7,200 seconds
origin_airport_window = Window.partitionBy('ORIGIN')\
                              .orderBy(unix_timestamp('dep_datetime_scheduled_utc'))\
                              .rangeBetween(-7200, 7200)

df_airlines = df_airlines.withColumn('DEP_CNT', count(col('dep_datetime_scheduled_utc')).over(origin_airport_window))


# number of flights arrivals scheduled between t-1 and t+1 at destination airport
# 1 hour before: -7,200 seconds. 1 hour after: 7,200 seconds
dest_airport_window = Window.partitionBy('DEST')\
                            .orderBy(unix_timestamp('arriv_datetime_scheduled_utc'))\
                            .rangeBetween(-7200, 7200)

df_airlines = df_airlines.withColumn('DST_CNT', count(col('arriv_datetime_scheduled_utc')).over(dest_airport_window))

# COMMAND ----------

# MAGIC %md ### Hour of the flight
# MAGIC Extract the hour of the flight from scheduled departure timestamp. As we saw in the EDA, the hour of departure appears to be correlated with the outcome variable delay.

# COMMAND ----------

df_airlines = df_airlines.withColumn('H_DEP', hour('dep_datetime_scheduled'))

# COMMAND ----------

# MAGIC %md ### PageRank and proxy for airports categorical variable
# MAGIC 
# MAGIC The PageRank feature measures the importance of each node (in this case airport) within the graph, based on the number incoming relationships (incoming routes from other airports) and the importance of the corresponding source nodes (other airports). The logic behind this feature is that if an airport with many connections is delayed, this delay will likely propagate to other airports. 
# MAGIC 
# MAGIC Additionally, using information about the degree of an airport, we calculate a variable that represent the size of an airport to serve as a proxy for origin and destination airport.
# MAGIC 
# MAGIC PageRank notebook: https://adb-6759024569771990.10.azuredatabricks.net/?o=6759024569771990#notebook/1310479932810734/command/1407650788079615

# COMMAND ----------

# Load PageRank, degree, an proxy airport data

df_pagerank = spark.read.parquet("/mnt/Azure/processed_data/pagerank_degree.parquet")

print(f'{df_pagerank.count():,} nodes loaded')

# Join with airlines dataframe on ORIGIN
df_airlines1 = df_airlines.join(broadcast(df_pagerank), df_airlines.ORIGIN == df_pagerank.id, 'left')

# list of columns to keep
keep_columns = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', \
                'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME',\
                'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID',\
                'DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', \
                'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF',\
                'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK',\
                'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', \
                'carrier_delay', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_datetime_scheduled', 'dep_datetime_actual',\
                'arriv_datetime_scheduled', 'arriv_datetime_actual', 'station_origin', 'airp_origin_timezone', 'station_origin_timezone',\
                'station_dest', 'airp_dest_timezone', 'station_dest_timezone', 'dep_datetime_scheduled_utc', 'dep_datetime_actual_utc', \
                'arriv_datetime_scheduled_utc', 'arriv_datetime_actual_utc', 'airport_delay', 'cutoff_time_utc', 'dep_delay_aircraft', 'flight_aircraft', \
                'DEP_CNT', 'DST_CNT', 'pagerank', 'degree', 'airport_size_group','H_DEP']

# keep relevant columns 
df_airlines1 = df_airlines1.select(*keep_columns)

# rename columns 
df_airlines1 = df_airlines1.withColumnRenamed('degree','degree_origin') \
                           .withColumnRenamed('airport_size_group','proxy_origin') \
                           .withColumnRenamed('pagerank','ORPageRank')



# COMMAND ----------

# Join with airlines dataframe on DEST to get degree and airport proxy on destination
df_airlines2 = df_airlines1.join(broadcast(df_pagerank), df_airlines1.DEST == df_pagerank.id, 'left')

# list of columns to keep
keep_columns2 = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', \
                'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME',\
                'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID',\
                'DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', \
                'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF',\
                'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK',\
                'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', \
                'carrier_delay', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_datetime_scheduled', 'dep_datetime_actual',\
                'arriv_datetime_scheduled', 'arriv_datetime_actual', 'station_origin', 'airp_origin_timezone', 'station_origin_timezone',\
                'station_dest', 'airp_dest_timezone', 'station_dest_timezone', 'dep_datetime_scheduled_utc', 'dep_datetime_actual_utc', \
                'arriv_datetime_scheduled_utc', 'arriv_datetime_actual_utc', 'airport_delay', 'cutoff_time_utc', 'dep_delay_aircraft', 'flight_aircraft', \
                'DEP_CNT', 'DST_CNT', 'ORPageRank', 'degree_origin', 'proxy_origin', 'pagerank', 'degree', 'airport_size_group','H_DEP' ]

# keep relevant columns 
df_airlines2 = df_airlines2.select(*keep_columns2)

# rename columns 
df_airlines2 = df_airlines2.withColumnRenamed('degree','degree_dest') \
                           .withColumnRenamed('airport_size_group','proxy_dest') \
                           .withColumnRenamed('pagerank','DESTPageRank')


# COMMAND ----------

# MAGIC %md ### Weather features
# MAGIC In this section we join the airlines table and the weather table. For each record in the airlines table (each flight) we include weather observations at origin and destination airports. 

# COMMAND ----------

df_weather.registerTempTable('df_weather')
df_airlines2.registerTempTable('df_airlines2')

df_final = sqlContext.sql(""" 

--Airlines
Select AL1.*, 

--Source weather
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

--Destination weather
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
AL3.GA1_5 As Dst_GA1_5

from df_airlines2 AL1
LEFT JOIN df_weather AL2 ON 
      AL1.station_origin = AL2.STATION 
  AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL2.DATE_JOIN_2H
LEFT JOIN df_weather AL3 ON 
      AL1.station_dest = AL3.STATION 
  AND to_timestamp(concat(YEAR(dep_datetime_scheduled),'-',MONTH(dep_datetime_scheduled),'-',DAY(dep_datetime_scheduled),'T',HOUR(dep_datetime_scheduled),':','00',':00.000+0000'))  = AL3.DATE_JOIN_2H

""")

# COMMAND ----------

# select columns to keep
columns_keep =   ['DAY_OF_MONTH', 'DAY_OF_WEEK','YEAR', 'QUARTER', 'MONTH', 'H_DEP', 'ORPageRank', 'proxy_origin', 'DESTPageRank', 'proxy_dest','OP_CARRIER',
                  'carrier_delay','airport_delay', 'DISTANCE', 'flight_aircraft', 'dep_delay_aircraft', 'DEP_CNT', 'DST_CNT', 'DEP_DEL15', 'Src_WND_0', 
                  'Src_WND_1', 'Src_WND_3', 'Src_WND_4', 'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1', 'Src_VIS_2', 'Src_VIS_3', 'Src_TMP_0', 'Src_TMP_1', 
                  'Src_DEW_0', 'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 
                  'Dst_WND_0', 'Dst_WND_1', 'Dst_WND_3', 'Dst_WND_4', 'Dst_CIG_0', 'Dst_CIG_1', 'Dst_VIS_0', 'Dst_VIS_1', 'Dst_VIS_2', 'Dst_VIS_3', 
                  'Dst_TMP_0', 'Dst_TMP_1', 'Dst_DEW_0', 'Dst_DEW_1', 'Dst_SLP_0', 'Dst_SLP_1', 'Dst_GA1_0', 'Dst_GA1_1', 'Dst_GA1_2', 'Dst_GA1_3', 
                  'Dst_GA1_4', 'Dst_GA1_5',    'FL_DATE', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'CRS_ARR_TIME', 'ORIGIN', 'DEST_AIRPORT_ID']

df_final = df_final.select(*columns_keep)

# COMMAND ----------

# MAGIC %md ### Check for null values
# MAGIC In this section we handle null values in the data.

# COMMAND ----------

# check for null values 
df_final.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_final.columns]).toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC - The variable `carrier_delay` has null values. These null values appear when there is no information in the -14h:15m-2h:15m window to calculate the delay of the carrier. In these cases, we will assume no delay and replace the null values by zero.
# MAGIC - The variable `airport_delay` has null values. We treat these cases in the same way as above. We will assume no delay and replace the null values by zero.
# MAGIC - The variable `flight_aircraft` has null values. These null values are because there are records with no aircraft tail number, and this variable is based on the aircraft tail number. These records are going to be filtered out. 
# MAGIC - The variable `dep_delay_aircraft` has null values. These values also have their roots in the absence of aircraft tail number. In some cases, is because there is no information available of previous departures in the time window considered. As we are going to filter out records with no aircraft tail number information many of these null values are going to disappear. For the remaining records with null values we assume no delay, hence we replace the null by zero.
# MAGIC - The weather variables have null values. These null values are going to be replaced by calculated average values.

# COMMAND ----------

# Fix null values

print(f'Number of records before fixing null values: {df_final.count()}')

df_final = df_final.na.fill(value=0,subset=['carrier_delay'])
df_final = df_final.na.fill(value=0,subset=['airport_delay'])
df_final = df_final.where(col('flight_aircraft').isNotNull())
df_final = df_final.na.fill(value=0,subset=['dep_delay_aircraft'])

df_final = df_final.na.fill(value=191.086034312939,subset=['Src_WND_0'])
df_final = df_final.na.fill(value=5.72357620777156,subset=['Src_WND_1'])
df_final = df_final.na.fill(value=32.4866334329149,subset=['Src_WND_3'])
df_final = df_final.na.fill(value=4.76868518022325,subset=['Src_WND_4'])
df_final = df_final.na.fill(value=14187.9922049176,subset=['Src_CIG_0'])
df_final = df_final.na.fill(value=5.34314229485104,subset=['Src_CIG_1'])
df_final = df_final.na.fill(value=14861.7568941402,subset=['Src_VIS_0'])
df_final = df_final.na.fill(value=5.18320388944515,subset=['Src_VIS_1'])
df_final = df_final.na.fill(value=9,subset=['Src_VIS_2'])
df_final = df_final.na.fill(value=5.83272417230157,subset=['Src_VIS_3'])
df_final = df_final.na.fill(value=128.150652427398,subset=['Src_TMP_0'])
df_final = df_final.na.fill(value=4.3669120773439,subset=['Src_TMP_1'])
df_final = df_final.na.fill(value=63.7693705711082,subset=['Src_DEW_0'])
df_final = df_final.na.fill(value=5.09696044502617,subset=['Src_DEW_1'])
df_final = df_final.na.fill(value=10161.8193223597,subset=['Src_SLP_0'])
df_final = df_final.na.fill(value=6.76090483370341,subset=['Src_SLP_1'])
df_final = df_final.na.fill(value=3.45184787386221,subset=['Src_GA1_0'])
df_final = df_final.na.fill(value=5.03447864378367,subset=['Src_GA1_1'])
df_final = df_final.na.fill(value=1362.80313152472,subset=['Src_GA1_2'])
df_final = df_final.na.fill(value=6.93450578301615,subset=['Src_GA1_3'])
df_final = df_final.na.fill(value=98.9195925988655,subset=['Src_GA1_4'])
df_final = df_final.na.fill(value=8.99639637985335,subset=['Src_GA1_5'])

df_final = df_final.na.fill(value=191.086034312939,subset=['Dst_WND_0'])
df_final = df_final.na.fill(value=5.72357620777156,subset=['Dst_WND_1'])
df_final = df_final.na.fill(value=32.4866334329149,subset=['Dst_WND_3'])
df_final = df_final.na.fill(value=4.76868518022325,subset=['Dst_WND_4'])
df_final = df_final.na.fill(value=14187.9922049176,subset=['Dst_CIG_0'])
df_final = df_final.na.fill(value=5.34314229485104,subset=['Dst_CIG_1'])
df_final = df_final.na.fill(value=14861.7568941402,subset=['Dst_VIS_0'])
df_final = df_final.na.fill(value=5.18320388944515,subset=['Dst_VIS_1'])
df_final = df_final.na.fill(value=9,subset=['Dst_VIS_2'])
df_final = df_final.na.fill(value=5.83272417230157,subset=['Dst_VIS_3'])
df_final = df_final.na.fill(value=128.150652427398,subset=['Dst_TMP_0'])
df_final = df_final.na.fill(value=4.3669120773439,subset=['Dst_TMP_1'])
df_final = df_final.na.fill(value=63.7693705711082,subset=['Dst_DEW_0'])
df_final = df_final.na.fill(value=5.09696044502617,subset=['Dst_DEW_1'])
df_final = df_final.na.fill(value=10161.8193223597,subset=['Dst_SLP_0'])
df_final = df_final.na.fill(value=6.76090483370341,subset=['Dst_SLP_1'])
df_final = df_final.na.fill(value=3.45184787386221,subset=['Dst_GA1_0'])
df_final = df_final.na.fill(value=5.03447864378367,subset=['Dst_GA1_1'])
df_final = df_final.na.fill(value=1362.80313152472,subset=['Dst_GA1_2'])
df_final = df_final.na.fill(value=6.93450578301615,subset=['Dst_GA1_3'])
df_final = df_final.na.fill(value=98.9195925988655,subset=['Dst_GA1_4'])
df_final = df_final.na.fill(value=8.99639637985335,subset=['Dst_GA1_5'])

print(f'Number of records after fixing null values: {df_final.count()}')

# COMMAND ----------

# Check again for null values to verify there are no null values
df_final.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_final.columns]).toPandas()

# COMMAND ----------

# MAGIC %md No null values in the dataframe. We can now save to a parquet file. 
# MAGIC 
# MAGIC 
# MAGIC Note: The only variable that shows null values (`DEP_TIME`) is not used as variable in our models. It was left in the dataframe to try some encoding methods in the toy dataset.

# COMMAND ----------

# save flights data with all features including weather

dbutils.fs.rm(path + "/processed_data/" + "airlines_with_features.parquet", recurse =True)
df_final.write.format("parquet").save(path + "/processed_data/" + "airlines_with_features.parquet")

display(dbutils.fs.ls(path + "/processed_data/") )

# COMMAND ----------

# create temp view
df_final.createOrReplaceTempView("df_final")

# COMMAND ----------

# MAGIC %sql DROP TABLE IF EXISTS airlines_with_features_sql; CREATE TABLE airlines_with_features_sql AS SELECT * FROM df_final

# COMMAND ----------

# MAGIC %md ## Modeling
# MAGIC 
# MAGIC In this section we create our baseline model, develop more sophisticated models and experiment with methods to improve model's performance: oversampling, PCA among others.

# COMMAND ----------

# MAGIC %md ### Helper functions

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

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

# MAGIC %md ### Get the data

# COMMAND ----------

#SET CONTEXT
sc = spark.sparkContext
sqlContext = SQLContext(sc)

#GET DATA
df = sqlContext.sql("""SELECT * from airlines_with_features_sql""")

# COMMAND ----------

print(get_dtype(df,'int'))
print("================================")
print(get_dtype(df,'double'))
print("================================")
print(get_dtype(df,'long'))
print("================================")
print(get_dtype(df,'string'))

# COMMAND ----------

# MAGIC %md ### Model data pipeline
# MAGIC Pipeline to prepare data for modeling.

# COMMAND ----------

#SELECT FEATURES - NUMERIC
numericCols = ['DAY_OF_MONTH', 'DAY_OF_WEEK','YEAR', 'QUARTER', 'MONTH', 'H_DEP', 'ORPageRank', 'proxy_origin', 'DESTPageRank', 'proxy_dest',
               #'carrier_delay','airport_delay', 'DISTANCE', 'flight_aircraft', 'dep_delay_aircraft', 
               'DISTANCE', 'DEP_CNT', 'DST_CNT', 'Src_WND_0', 'Src_WND_1', 
               'Src_WND_3', 'Src_WND_4', 'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1', 'Src_VIS_2', 'Src_VIS_3', 'Src_TMP_0', 'Src_TMP_1', 
               'Src_DEW_0', 'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 
               'Dst_WND_0', 'Dst_WND_1', 'Dst_WND_3', 'Dst_WND_4', 'Dst_CIG_0', 'Dst_CIG_1', 'Dst_VIS_0', 'Dst_VIS_1', 'Dst_VIS_2', 'Dst_VIS_3', 
               'Dst_TMP_0', 'Dst_TMP_1', 'Dst_DEW_0', 'Dst_DEW_1', 'Dst_SLP_0', 'Dst_SLP_1', 'Dst_GA1_0', 'Dst_GA1_1', 'Dst_GA1_2', 'Dst_GA1_3', 
               'Dst_GA1_4', 'Dst_GA1_5']

#SELECT FEATURES - CATEGORICAL
categoricalColumns = ['OP_CARRIER']

# COMMAND ----------

# MAGIC %md Below we create a pipeline to prepare the data for modeling: 
# MAGIC - Create indexer for categorical and numerical variables.
# MAGIC - Create indexer for the label column.
# MAGIC - One-hot encode categorical variable.
# MAGIC - Create one column that contains a vector with all features of the model.

# COMMAND ----------

# List of columns in the dataframe
cols = df.columns

# Create indexer for categorical variables and encode them using one-hot-encoding
# Add these steps to the pipeline stages
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder (inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
    
# Create indexer for the label variable
# Add this step to the pipeline stages
label_stringIdx = StringIndexer(inputCol = 'DEP_DEL15', outputCol = 'label')
stages += [label_stringIdx]

# Create a transformer that merges multiple columns into a vector column that represents the features for the model
# Add this step to the pipeline stages
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Create a pipeline: sequence of stages to transform the data
pipeline = Pipeline(stages = stages)

# The fit() method is called to produce a transformer that contains all transformations in stages
pipelineModel = pipeline.fit(df)

# Run the transformation in the dataframe
df = pipelineModel.transform(df)

# Select columns including the labels and the features column with all the features as a vector
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)

# COMMAND ----------

# MAGIC %md ### Split data into training, development and test sets
# MAGIC 
# MAGIC All the data but the last 3 months is used for training. The last 3 months of 2019 are split into test and development sets to run prediction on more recent data. This represents a rolling window where the model is always tested with the most recent data and trained with all remaining data. 

# COMMAND ----------

# TRAIN AND ALL DATA BUT LAST 3M
train = df.where(((col('YEAR') == 2015) | (col('YEAR') == 2016) | (col('YEAR') == 2017) | (col('YEAR') == 2018)) | ((col('YEAR') == 2019) & (col('MONTH') <10) ))
print(f'{train.count():} records in train data')

# TEST/DEV ON LAST 3M OF DATA
test, dev = (df.where((col('YEAR') == 2019) & (col('MONTH')>=10))).randomSplit([0.5,0.5],seed=1)
print(f'{test.count():} records in test data')  
print(f'{dev.count():} records in dev data')

# COMMAND ----------

# MAGIC %md ### Baseline: Logistic Regression
# MAGIC 
# MAGIC For its simplicity, as a baseline algorithm we chose to use Logistic Regression, which is a very common algorithm and probably one of the most common tools for classification. 
# MAGIC 
# MAGIC In the `Algorithm Implementation` section there is a detailed description of the Logistic Regression algorithm. 

# COMMAND ----------

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)

# COMMAND ----------

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set area Under ROC: ' + str(trainingSummary.areaUnderROC))

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

class_temp = predictions.select("label").groupBy("label").count().sort('count', ascending=False).toPandas()
class_temp = class_temp["label"].values.tolist()
class_names = map(str, class_temp)
print(class_names)

# Get Predicted VS Actual
y_true = predictions.select("label").toPandas()
y_pred = predictions.select("prediction").toPandas()
cnf_matrix = confusion_matrix(y_true, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['no_delay','delay']
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

print(accuracy_score(y_true, y_pred))

# COMMAND ----------

# Get Predicted VS Actual
y_true = predictions.select("label").toPandas()
y_pred = predictions.select("prediction").toPandas()
cnf_matrix = confusion_matrix(y_true, y_pred)


# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# COMMAND ----------

# MAGIC %md The departure delay prediction is very complex and non-linear. Because of this, a linear classifier such as logistic regression does not achieve high performance. Only 11% of departure delays are detected, and the f1-score is low (0.19).

# COMMAND ----------

# MAGIC %md ### Sophisticated Modeling
# MAGIC 
# MAGIC In this section we develop models that can better handle the non-linear relationship between our predictors and the outcome variable.

# COMMAND ----------

#SET CONTEXT
sc = spark.sparkContext
sqlContext = SQLContext(sc)

#GET DATA
df = sqlContext.sql("""SELECT * FROM airlines_with_features_sql""")

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

#SELECT FEATURES - NUMERIC ALL
numericColsAll =   ['ORPageRank', 'proxy_origin', 'proxy_dest','carrier_delay','airport_delay','H_DEP','Src_WND_0', 'Src_WND_1', 'Src_WND_3', 'Src_WND_4', 'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1', 'Src_VIS_2', 'Src_VIS_3', 'Src_TMP_0', 'Src_TMP_1', 'Src_DEW_0', 'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 'Dst_WND_0', 'Dst_WND_1', 'Dst_WND_3', 'Dst_WND_4', 'Dst_CIG_0', 'Dst_CIG_1', 'Dst_VIS_0', 'Dst_VIS_1', 'Dst_VIS_2', 'Dst_VIS_3', 'Dst_TMP_0', 'Dst_TMP_1', 'Dst_DEW_0', 'Dst_DEW_1', 'Dst_SLP_0', 'Dst_SLP_1', 'Dst_GA1_0', 'Dst_GA1_1', 'Dst_GA1_2', 'Dst_GA1_3', 'Dst_GA1_4', 'Dst_GA1_5','DEP_CNT', 'DST_CNT', 'DESTPageRank', 'DAY_OF_MONTH', 'DAY_OF_WEEK','YEAR', 'QUARTER', 'MONTH' , 'DISTANCE','flight_aircraft','dep_delay_aircraft']

#SELECT FEATURES - NOT WEATHER
numericCols =   ['ORPageRank', 'proxy_origin', 'proxy_dest','carrier_delay','airport_delay','H_DEP','DEP_CNT', 'DST_CNT', 'DESTPageRank', 'DAY_OF_MONTH', 'DAY_OF_WEEK','YEAR', 'QUARTER', 'MONTH' , 'DISTANCE','flight_aircraft','dep_delay_aircraft']

#SELECT FEATURES WEATHER 
numericColsWH =   ['Src_WND_0', 'Src_WND_1', 'Src_WND_3', 'Src_WND_4', 'Src_CIG_0', 'Src_CIG_1', 'Src_VIS_0', 'Src_VIS_1', 'Src_VIS_2', 'Src_VIS_3', 'Src_TMP_0', 'Src_TMP_1', 'Src_DEW_0', 'Src_DEW_1', 'Src_SLP_0', 'Src_SLP_1', 'Src_GA1_0', 'Src_GA1_1', 'Src_GA1_2', 'Src_GA1_3', 'Src_GA1_4', 'Src_GA1_5', 'Dst_WND_0', 'Dst_WND_1', 'Dst_WND_3', 'Dst_WND_4', 'Dst_CIG_0', 'Dst_CIG_1', 'Dst_VIS_0', 'Dst_VIS_1', 'Dst_VIS_2', 'Dst_VIS_3', 'Dst_TMP_0', 'Dst_TMP_1', 'Dst_DEW_0', 'Dst_DEW_1', 'Dst_SLP_0', 'Dst_SLP_1', 'Dst_GA1_0', 'Dst_GA1_1', 'Dst_GA1_2', 'Dst_GA1_3', 'Dst_GA1_4', 'Dst_GA1_5']

#SELECT FEATURES - CATEGORICAL
categoricalColumns = ['OP_CARRIER']

# COMMAND ----------

# MAGIC %md #### Principal Component Analysis (PCA)
# MAGIC 
# MAGIC PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. 
# MAGIC 
# MAGIC We have identified the weather features as an ideal feature set to use with PCA due to the highly correlated of the many measurments that make up each observation. 
# MAGIC 
# MAGIC Components Number is set at 6 based on running PCA with a larger size and using the Explained Variance graph to determine that no additional variance is added if a larger number is used

# COMMAND ----------

#VECTOR THE INPUT DATA
assembler = VectorAssembler(inputCols=numericColsWH, outputCol="PCA_IN_Numfeatures")
df = assembler.transform(df)

#RUN PCA
pca = PCA(k=7 , inputCol="PCA_IN_Numfeatures" , outputCol="PCA_OUT_Numfeatures")
model = pca.fit(df)
df = model.transform(df)

#GRAPH EXPLAINED VARIANCE
data = model.explainedVariance.values
plt.bar(range(1,8,1), data)
plt.xticks(np.arange(1, 8, 1))
plt.title("Principal Component Explained Variance VS Each Principal Component")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")

# COMMAND ----------

# MAGIC %md #### Model Data PipeLine

# COMMAND ----------

#ALL COLUMNS
cols = df.columns

#VECTORIZE NUMERIC COLS
assembler = VectorAssembler(inputCols=numericCols, outputCol="Numfeatures")
df = assembler.transform(df)

#MIN-MAX-SCALER
Scaler = MinMaxScaler(inputCol="Numfeatures", outputCol="scaledFeatures")
df = Scaler.fit(df).transform(df)

#CAT COLUMNS
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder (inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
    
#LABELS
label_stringIdx = StringIndexer(inputCol = 'DEP_DEL15', outputCol = 'label')
stages += [label_stringIdx]

#ASSEBLER
assemblerInputs = [c + "classVec" for c in categoricalColumns] 
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="featuresCat")
stages += [assembler]

#PIPELINE
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'featuresCat'] + cols + ['Numfeatures','scaledFeatures']
df = df.select(selectedCols)

#ADD CAT_FEATURES & MINMAX & PCA --> VECTOR
assembler = VectorAssembler(inputCols=["featuresCat", "scaledFeatures","PCA_OUT_Numfeatures"],outputCol="features")
df = assembler.transform(df)

# COMMAND ----------

# TRAIN AND ALL DATA BUT LAST 3M
train = df.where(((col('YEAR') == 2015) | (col('YEAR') == 2016) | (col('YEAR') == 2017) | (col('YEAR') == 2018)) | ((col('YEAR') == 2019) & (col('MONTH') <10) ))
print(f'{train.count():} records in train data')

# TEST/DEV ON LAST 3M OF DATA
test, dev = (df.where((col('YEAR') == 2019) & (col('MONTH')>=10))).randomSplit([0.5,0.5],seed=1)
print(f'{test.count():} records in test data')  
print(f'{dev.count():} records in dev data')

# OVERSAMPLE MINORITY CLASS
train = sample_df(train,ratio,1)

# COMMAND ----------

# MAGIC %md #### LogisticRegression
# MAGIC 
# MAGIC Hyperparameters: 
# MAGIC 
# MAGIC Elastic net contains both L1 and L2 regularization as special cases. If elastic net parameter α set to 1, it is equivalent to a Lasso model. On the other hand, if α is set to 0, the trained model reduces to a ridge regression model.
# MAGIC 
# MAGIC - elasticNetParam corresponds to α 
# MAGIC - regParam corresponds to λ.

# COMMAND ----------

#regParam=0.1, elasticNetParam=0.95
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10, family="multinomial")
lrModel = lr.fit(train)
predictions = lrModel.transform(test)

# COMMAND ----------

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set area Under ROC: ' + str(trainingSummary.areaUnderROC))

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

class_temp = predictions.select("label").groupBy("label").count().sort('count', ascending=False).toPandas()
class_temp = class_temp["label"].values.tolist()
class_names = map(str, class_temp)
print(class_names)

# Get Predicted VS Actual
y_true = predictions.select("label").toPandas()
y_pred = predictions.select("prediction").toPandas()
cnf_matrix = confusion_matrix(y_true, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['no_delay','delay']
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

print(accuracy_score(y_true, y_pred))

# COMMAND ----------

# MAGIC %md #### DecisionTree

# COMMAND ----------

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=6)
 
# Train model with Training Data
dtModel = dt.fit(train)

# COMMAND ----------

print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)
display(dtModel)

# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(test)

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md #### RandomForestClassifier

# COMMAND ----------

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees = 100)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

ExtractFeatureImp(rfModel.featureImportances, train, "features").head(100)

# COMMAND ----------

# MAGIC %md #### GBTClassifier

# COMMAND ----------

gbt = GBTClassifier(maxIter=100)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

# MAGIC %md ## Conclusions
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From project instructions: report results and learnings for both the ML as well as the scalability.
# MAGIC >
# MAGIC > From rubric:
# MAGIC > - Interpret and visualize the model scores, ie: confusion matrix.
# MAGIC > - Draw effective conclusions that justify that the problem as envisioned originally is actually solved.

# COMMAND ----------

# MAGIC %md ## Algorithm Implementation
# MAGIC 
# MAGIC We implemented Logistic Regression. 
# MAGIC 
# MAGIC The algorithm implementation is available in the notebook:

# COMMAND ----------

# MAGIC %md ## Application of Course Concepts

# COMMAND ----------

# MAGIC %md ###Encoding
# MAGIC 
# MAGIC The airline/weather dataset has multiple fields that are numerical as well categorical in nature. Most machine learning algorithms do much better with numerical data than categorical data. While MLlib does account for some these conversions in its algorithm implmentations, doing encoding on the data prior to feeding it into an algorithm is advantageous as it gives the user the ability to choose the encoding method they may think is more appropriate to the dataset. For example, while One Hot Encoding could be leveraged, high cardinality on the data would result in larger set of features getting fed into the algorithm. This naturally can increase computation time as well as overall complexity of the model. At the end of the day, whether something like Feature hashing or Target encoding is applied, will really be based on general performance of the model with that encoding scheme applied. 

# COMMAND ----------

# MAGIC %md ###Scaling
# MAGIC 
# MAGIC A lot of the numerical data, more specifically the weather data has a varied range of continuous values. What one needs to avoid is have an algorithm give more weightage to a feature due to its higher ranging numbers. One other reason here is for algorithms like Logistic regression, gradient descent converges faster with scaled features than without - a case of saturation on an activation function.
# MAGIC 
# MAGIC Another factor to consider here is in regards to feature selection. For leveraging PCA on a large dataset with a lot of features, scaling becomes critical for better performance given that features are selected based on maximum variance and magnitude. Naturally, unscaled data skews PCA results as well.

# COMMAND ----------

# MAGIC %md ###PageRank
# MAGIC 
# MAGIC A feature that we introduced into the dataset was in noting the importance of an airport as hub in the overall domestic airport network. An airport's importance correlates to its impact on flight operations in terms of throughput of traffic. An important, but congested airport, may become responsible for more delayed flights. This effect is important to capture, and Pagerank can be used in a meaningful way. Calculating the degree centrality of an airport, meaning the number of routes (edges) connected to an airport depict its centrality measure in the ariport network, and thus its importance.

# COMMAND ----------

# MAGIC %md ## References
# MAGIC 
# MAGIC 1. A Data Mining Approach to Flight Arrival Delay Prediction for American Airlines: https://www.researchgate.net/publication/331858316_A_Data_Mining_Approach_to_Flight_Arrival_Delay_Prediction_for_American_Airlines
# MAGIC 2. Airport’s location dataset: https://data.humdata.org/dataset/ourairports-usa
# MAGIC 3. Dictionary for airlines dataset: https://www.transtats.bts.gov/DL_SelectFields.asp?gnoyr_VQ=FGJ
# MAGIC 4. Dictionary for weather dataset: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# MAGIC 5. Oversampling/Undersampling: https://medium.com/@junwan01/oversampling-and-undersampling-with-pyspark-5dbc25cdf253

# COMMAND ----------


