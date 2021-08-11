# Databricks notebook source
# MAGIC %md # Flight departure delay predictions
# MAGIC ## Extract, transform, load (ETL)
# MAGIC 
# MAGIC Team members:
# MAGIC - Isabel Garcia Pietri
# MAGIC - Madhu Hegde
# MAGIC - Amit Karandikar
# MAGIC - Piotr Parkitny
# MAGIC 
# MAGIC This notebook contains the ETL pipeline we use to pre-process airlines and weather data. Two parquet files are generated as a result of this pipeline: one for the weather data and another for the airlines data. These files are ready to be joined. We perform this join in the Feature Engineering section to incorporate weather features to our main dataframe.

# COMMAND ----------

# MAGIC %md ## Package imports, directories and configuration

# COMMAND ----------

# install package to find timezone based on longitude and latitude
%pip install timezonefinder

# COMMAND ----------

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, expr, unix_timestamp, avg, round, lag, to_date, row_number, explode, array, lit
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType

import numpy as np
import math
from timezonefinder import TimezoneFinder


# COMMAND ----------

#data path to our directory 
path = 'dbfs:/mnt/Azure/'
display(dbutils.fs.ls(path))

# COMMAND ----------

# Inspect directory where the original data is
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# spark configuration
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md ## User defined functions
# MAGIC 
# MAGIC We define two UDFs: 
# MAGIC 
# MAGIC - Function that calculates the shortest distance between two points given their latitude and longitude. This function is used to calculate the weather station closest to an airport.
# MAGIC - Function that returns the time zone of a point given latitude and longitude. This function is used to convert timestamps to UTC.

# COMMAND ----------

# define a udf to calculate distance given two points of latitude and longitude
@udf("double")
def haversine(origin_lat:float, origin_long: float, dest_lat: float, dest_long:float):
  
  R = 3958.8 # in miles
  # compute
  rlat1 = origin_lat * (math.pi/180)
  rlat2 = dest_lat * (math.pi/180)
  diff_lat = rlat2 - rlat1
  diff_long = (dest_long * (math.pi/180)) - (origin_long * (math.pi/180))  
  d = 2 * R * math.asin(math.sqrt(math.sin(diff_lat/2)*math.sin(diff_lat/2) + \
                                  math.cos(rlat1)*math.cos(rlat2)*math.sin(diff_long/2)*math.sin(diff_long/2)))
  
  # returns distance in miles
  return d

# register the function for use in sql
spark.udf.register("haversine", haversine)

# define a udf to calculate the timezone 
@udf("string")
def timezone(latitude:float, longitude: float):
  
  tf = TimezoneFinder()
  timezone = tf.timezone_at(lng=longitude, lat=latitude) 
  
  # returns timezone
  return timezone

# register the function for use in sql
spark.udf.register("timezone", timezone)

# COMMAND ----------

# MAGIC %md ## Airline's data
# MAGIC 
# MAGIC The airline's dataset has 31,746,841 unique records. In the original data every record is duplicated, but all duplications are removed when the data is loaded.
# MAGIC 
# MAGIC This dataset includes flights cancelled for many reasons: carrier caused, weather, national aviation system and security. These flights could be considered as extreme delays, where passengers are relocated to other flights. For the purpose of this project, we consider that cancelled flights are delayed flights with a delay of 2 hours, which is an approximate of the time it takes to take another flight within the US.
# MAGIC 
# MAGIC Similarly, this dataset contains diverted flights. These flights are also kept, as our main focus in this study is to analyze departure delays. Flight diversions happen after departure and can be caused by many reasons unrelated with this study: aircraft emergency, passenger emergency, mechanical failure, etc.
# MAGIC 
# MAGIC We found some instances were variables that measure delay at departure and arrival were null for no reason. These cases were present when the flight departed/arrived at the exact scheduled time. These cases were fixed to reflect no delay. 
# MAGIC 
# MAGIC Few records (~5) had null values in our outcome variable (DEP_DEL15), these records were removed.
# MAGIC 
# MAGIC Additionally, in this section dates/times are converted to datetime objects and irrelevant columns are filtered out (e.g., columns related with diversion airports).

# COMMAND ----------

# Load airlines data and drop duplicates
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*").dropDuplicates()

# print number of records
print(f'Number of flight records loaded: {df_airlines.count()}')

# print number of columns
print(f'Number of columns loaded: {len(df_airlines.columns)}')

# Assume CANCELLED flights as delayed for 2 hours 
# set DEP_DELAY = 120 minutes
df_airlines = df_airlines.withColumn('DEP_DELAY', when(df_airlines.CANCELLED == 1, 120 )\
                                     .otherwise(df_airlines.DEP_DELAY))

# set DEP_DELAY_NEW = 120 minutes
df_airlines = df_airlines.withColumn('DEP_DELAY_NEW', when(df_airlines.CANCELLED == 1, 120)\
                                     .otherwise(df_airlines.DEP_DELAY_NEW))

# set indicator variable DEP_DEL15 to 1 
df_airlines = df_airlines.withColumn('DEP_DEL15', when(df_airlines.CANCELLED == 1, 1 )\
                                      .otherwise(df_airlines.DEP_DEL15))

# set indicator variable DEP_DELAY_GROUP to group=8 (group=8, delay between 120 and 134 minutes)
df_airlines = df_airlines.withColumn('DEP_DELAY_GROUP', when(df_airlines.CANCELLED == 1, 8 )\
                                      .otherwise(df_airlines.DEP_DELAY_GROUP))

# Fix the null values in the delay DEPARTURE variables, when the flight departed at the exact scheduled time.
# set DEP_DELAY = 0 to on-time flights
df_airlines = df_airlines.withColumn('DEP_DELAY', when(df_airlines.CRS_DEP_TIME == df_airlines.DEP_TIME, 0 )\
                                     .otherwise(df_airlines.DEP_DELAY))

# set DEP_DELAY_NEW = 0 to on-time flights
df_airlines = df_airlines.withColumn('DEP_DELAY_NEW', when(df_airlines.CRS_DEP_TIME == df_airlines.DEP_TIME, 0 )\
                                     .otherwise(df_airlines.DEP_DELAY_NEW))

# set indicator variable DEP_DEL15 to 0 when flight is on-time
df_airlines = df_airlines.withColumn('DEP_DEL15', when(df_airlines.CRS_DEP_TIME == df_airlines.DEP_TIME, 0 )\
                                      .otherwise(df_airlines.DEP_DEL15))

# set indicator variable DEP_DELAY_GROUP to group=0 when flight is on-time (group=0, delay between 0 and 14 minutes)
df_airlines = df_airlines.withColumn('DEP_DELAY_GROUP', when(df_airlines.CRS_DEP_TIME == df_airlines.DEP_TIME, 0 )\
                                      .otherwise(df_airlines.DEP_DELAY_GROUP))

# Fix the null values in the delay ARRIVAL variables, when the flight arrived at the exact scheduled time.
# set ARR_DELAY = 0 to on-time flights
df_airlines = df_airlines.withColumn('ARR_DELAY', when(df_airlines.CRS_ARR_TIME == df_airlines.ARR_TIME, 0 )\
                                     .otherwise(df_airlines.ARR_DELAY))

# set ARR_DELAY_NEW = 0 to on-time flights
df_airlines = df_airlines.withColumn('ARR_DELAY_NEW', when(df_airlines.CRS_ARR_TIME == df_airlines.ARR_TIME, 0 )\
                                     .otherwise(df_airlines.ARR_DELAY_NEW))

# set indicator variable ARR_DEL15 to 0 when flight is on-time
df_airlines = df_airlines.withColumn('ARR_DEL15', when(df_airlines.CRS_ARR_TIME == df_airlines.ARR_TIME, 0 )\
                                      .otherwise(df_airlines.ARR_DEL15))

# set indicator variable ARR_DELAY_GROUP to group=0 when flight is on-time (group=0, delay between 0 and 14 minutes)
df_airlines = df_airlines.withColumn('ARR_DELAY_GROUP', when(df_airlines.CRS_ARR_TIME == df_airlines.ARR_TIME, 0 )\
                                      .otherwise(df_airlines.ARR_DELAY_GROUP))

# Scheduled departure datetime column. First add a pad to departure time to create a four digits time (e.g. convert 745 to 0745 )
# after concatenate date and departure time
df_airlines = df_airlines.withColumn('dep_time_scheduled', lpad(col('CRS_DEP_TIME'), 4, '0')) \
                         .withColumn('dep_datetime_scheduled', to_timestamp(concat(col('FL_DATE'), col('dep_time_scheduled')), format='yyyy-MM-ddHHmm'))

# Actual departure datetime column. First add a pad to departure time to create a four digits time (e.g. convert 745 to 0745 )
# after concatenate date and departure time
df_airlines = df_airlines.withColumn('dep_time_actual', lpad(col('DEP_TIME'), 4, '0')) \
                         .withColumn('dep_datetime_actual', to_timestamp(concat(col('FL_DATE'), col('dep_time_actual')), format='yyyy-MM-ddHHmm'))

# Scheduled arrival datetime column. First add a pad to arrival time to create a four digits time (e.g. convert 745 to 0745 )
# after concatenate date and departure time
df_airlines = df_airlines.withColumn('arriv_time_scheduled', lpad(col('CRS_ARR_TIME'), 4, '0')) \
                         .withColumn('arriv_datetime_scheduled', to_timestamp(concat(col('FL_DATE'), col('arriv_time_scheduled')), format='yyyy-MM-ddHHmm'))

# Actual arrival datetime column. First add a pad to arrival time to create a four digits time (e.g. convert 745 to 0745 )
# after concatenate date and departure time
df_airlines = df_airlines.withColumn('arriv_time_actual', lpad(col('ARR_TIME'), 4, '0')) \
                         .withColumn('arriv_datetime_actual', to_timestamp(concat(col('FL_DATE'), col('arriv_time_actual')), format='yyyy-MM-ddHHmm'))

# filter out flights that after all fixes remain with the DEP_DELAY15 variable null (~5 flights)
df_airlines = df_airlines.where(col('DEP_DEL15').isNotNull())

# Select columns to keep. Remove airport diversion related columns
airlines_columns = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', \
                    'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', \
                    'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', \
                    'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', \
                    'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', \
                    'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', \
                    'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', \
                    'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_datetime_scheduled', 'dep_datetime_actual', 'arriv_datetime_scheduled', \
                    'arriv_datetime_actual']

# keep relevant columns 
df_airlines = df_airlines.select(*airlines_columns)


# COMMAND ----------

# MAGIC %md ## Weather data
# MAGIC 
# MAGIC The Weather data is from the National Oceanic and Atmospheric Administration repository. 
# MAGIC 
# MAGIC Detail Explanation of all the fields can be found in the following document https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf. 
# MAGIC This dataset is a large collection of hourly and sub-hourly weather observations collected from around the world. 
# MAGIC 
# MAGIC The data includes the station description, latitude, longitude, and weather station IDs along with readings from many different sensors. 
# MAGIC 
# MAGIC Based on the EDA we will remove report_type = SOD and SOM which are daily and monthly summary.
# MAGIC 
# MAGIC The weather dataset is by far the largest that will be used. It contains 630,904,436 observations

# COMMAND ----------

# MAGIC %md ### Weather data: Part 1
# MAGIC  - As first step we will load the parquet file and filter to US based stations
# MAGIC  - We have selected certain columns which we will use in next steps for processing

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

# MAGIC %md ### Weather data: Part 2
# MAGIC 
# MAGIC This is the initial processing step of the weather data. The data is split into columns for processing in the next step.
# MAGIC - To be able to summarize the data, timestamp column is truncated on hour/day/minute
# MAGIC - All relevant fields are split into new fields
# MAGIC - We only use data where we have the station id
# MAGIC - REPORT_TYPE SOD and SOM is filtered out as those are daily and monthly summaries

# COMMAND ----------

df_weather_proc = sqlContext.sql('''

SELECT
   STATION
  ,DATE
  ,to_timestamp(concat(YEAR(DATE),'-',MONTH(DATE),'-',DAY(DATE),'T',HOUR(DATE),':',floor(minute(DATE)/10)*10,':00.000+0000')) AS DATE_JOIN_MN
  ,to_timestamp(concat(YEAR(DATE),'-',MONTH(DATE),'-',DAY(DATE),'T',HOUR(DATE),':','00',':00.000+0000')) AS DATE_JOIN_HH
  ,to_timestamp(concat(YEAR(DATE),'-',MONTH(DATE),'-',DAY(DATE),'T','00',':','00',':00.000+0000')) AS DATE_JOIN_DA
  
  ,SOURCE
  ,cast(LATITUDE  as float) as LATITUDE
  ,cast(LONGITUDE as float) as LONGITUDE
  ,cast(ELEVATION as float) as ELEVATION
  
  ,SPLIT(NAME,',')[0] as NAME_0
  ,SPLIT(NAME,',')[1] as NAME_1
  
  ,REPORT_TYPE
  ,CALL_SIGN
  ,QUALITY_CONTROL
  
  ,WND
  ,cast(SPLIT(WND,',')[0] as int) as WND_0
  ,cast(SPLIT(WND,',')[1] as int) as WND_1
  ,SPLIT(WND,',')[2]              as WND_2
  ,cast(SPLIT(WND,',')[3] as int) as WND_3
  ,cast(SPLIT(WND,',')[4] as int) as WND_4
  
  ,CIG
  ,cast(SPLIT(CIG,',')[0] as int) as CIG_0
  ,cast(SPLIT(CIG,',')[1] as int) as CIG_1
  ,SPLIT(CIG,',')[2]              as CIG_2
  ,SPLIT(CIG,',')[3]              as CIG_3
  
  ,VIS
  ,cast(SPLIT(VIS,',')[0] as int) as VIS_0
  ,cast(SPLIT(VIS,',')[1] as int) as VIS_1
  ,cast(SPLIT(VIS,',')[2] as int) as VIS_2
  ,cast(SPLIT(VIS,',')[3] as int) as VIS_3
  
  ,TMP
  ,IF(substring(TMP,0,1) = '-', (cast(substring(SPLIT(TMP,',')[0],2) as int) * -1), (cast(substring(SPLIT(TMP,',')[0],2) as int))) as TMP_0
  ,cast( SPLIT(TMP,',')[1] as int)              as TMP_1
  
  ,DEW  
  ,IF(substring(DEW,0,1) = '-', (cast(substring(SPLIT(DEW,',')[0],2) as int) * -1), (cast(substring(SPLIT(DEW,',')[0],2) as int))) as DEW_0  
  ,cast( SPLIT(DEW,',')[1] as int)              as DEW_1

  ,SLP
  ,cast( SPLIT(SLP,',')[0] as int)              as SLP_0
  ,cast( SPLIT(SLP,',')[1] as int)              as SLP_1

  ,AW1
  
  ,GA1
  ,cast( SPLIT(GA1,',')[0] as int)              as GA1_0
  ,cast( SPLIT(GA1,',')[1] as int)              as GA1_1    
  ,IF(substring(SPLIT(GA1,',')[2],0,1) = '-', (cast(substring(SPLIT(GA1,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA1,',')[2],2) as int))) as GA1_2  
  ,cast( SPLIT(GA1,',')[3] as int)              as GA1_3  
  ,cast( SPLIT(GA1,',')[4] as int)              as GA1_4  
  ,cast( SPLIT(GA1,',')[5] as int)              as GA1_5  
    
  ,GA2
  ,cast( SPLIT(GA2,',')[0] as int)              as GA2_0
  ,cast( SPLIT(GA2,',')[1] as int)              as GA2_1    
  ,IF(substring(SPLIT(GA2,',')[2],0,1) = '-', (cast(substring(SPLIT(GA2,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA2,',')[2],2) as int))) as GA2_2  
  ,cast( SPLIT(GA2,',')[3] as int)              as GA2_3  
  ,cast( SPLIT(GA2,',')[4] as int)              as GA2_4  
  ,cast( SPLIT(GA2,',')[5] as int)              as GA2_5  
  
  ,GA3
  ,cast( SPLIT(GA3,',')[0] as int)              as GA3_0
  ,cast( SPLIT(GA3,',')[1] as int)              as GA3_1    
  ,IF(substring(SPLIT(GA3,',')[2],0,1) = '-', (cast(substring(SPLIT(GA3,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA3,',')[2],2) as int))) as GA3_2    
  ,cast( SPLIT(GA3,',')[3] as int)              as GA3_3  
  ,cast( SPLIT(GA3,',')[4] as int)              as GA3_4  
  ,cast( SPLIT(GA3,',')[5] as int)              as GA3_5  
  
  ,GA4
  ,cast( SPLIT(GA4,',')[0] as int)              as GA4_0
  ,cast( SPLIT(GA4,',')[1] as int)              as GA4_1    
  ,IF(substring(SPLIT(GA4,',')[2],0,1) = '-', (cast(substring(SPLIT(GA4,',')[2],2) as int) * -1), (cast(substring(SPLIT(GA4,',')[2],2) as int))) as GA4_2    
  ,cast( SPLIT(GA4,',')[3] as int)              as GA4_3  
  ,cast( SPLIT(GA4,',')[4] as int)              as GA4_4  
  ,cast( SPLIT(GA4,',')[5] as int)              as GA4_5  
  
  ,GE1
  ,cast( SPLIT(GE1,',')[0] as int)              as GE1_0
  ,SPLIT(GE1,',')[1]                            as GE1_1    
  ,IF(substring(SPLIT(GE1,',')[2],0,1) = '-', (cast(substring(SPLIT(GE1,',')[2],2) as int) * -1), (cast(substring(SPLIT(GE1,',')[2],2) as int))) as GE1_2    
  ,IF(substring(SPLIT(GE1,',')[3],0,1) = '-', (cast(substring(SPLIT(GE1,',')[3],2) as int) * -1), (cast(substring(SPLIT(GE1,',')[3],2) as int))) as GE1_3  
  
  ,GF1
  ,cast( SPLIT(GF1,',')[0] as int)              as GF1_0
  ,cast( SPLIT(GF1,',')[1] as int)              as GF1_1
  ,cast( SPLIT(GF1,',')[2] as int)              as GF1_2
  ,cast( SPLIT(GF1,',')[3] as int)              as GF1_3
  ,cast( SPLIT(GF1,',')[4] as int)              as GF1_4
  ,cast( SPLIT(GF1,',')[5] as int)              as GF1_5
  ,cast( SPLIT(GF1,',')[6] as int)              as GF1_6
  ,cast( SPLIT(GF1,',')[7] as int)              as GF1_7
  ,cast( SPLIT(GF1,',')[8] as int)              as GF1_8
  ,cast( SPLIT(GF1,',')[9] as int)              as GF1_9
  ,cast( SPLIT(GF1,',')[10] as int)             as GF1_10
  ,cast( SPLIT(GF1,',')[11] as int)             as GF1_11
  ,cast( SPLIT(GF1,',')[12] as int)             as GF1_12
   
  ,KA1
  ,cast( SPLIT(KA1,',')[0] as int)              as KA1_0
  ,SPLIT(KA1,',')[1]                            as KA1_1
  ,IF(substring(SPLIT(KA1,',')[2],0,1) = '-', (cast(substring(SPLIT(KA1,',')[2],2) as int) * -1), (cast(substring(SPLIT(KA1,',')[2],2) as int))) as KA1_2B  
  ,cast( SPLIT(KA1,',')[3] as int)              as KA1_3 
FROM df_weather
WHERE STATION IS NOT NULL
AND REPORT_TYPE NOT IN ('SOD','SOM')
''' )

# COMMAND ----------

# MAGIC %md ### Weather data: Part 3
# MAGIC 
# MAGIC Summary Data is created using Dynamic SQL. The code allows to create summary data on requested columns by a custom group_col that can be expanded to any field we like.
# MAGIC 
# MAGIC Result from each sql query to added to a common table that can be used in the next step. 
# MAGIC 
# MAGIC To make sure that our prediction horizon is always under 2 hours we move back the clock on the weather data by 3 hours. This guarantees that the summary is always greater than 2 hours back from the airlines data.

# COMMAND ----------

df_weather_proc.createOrReplaceTempView("df_weather_proc")

sum_var = ['WND_0', 'WND_1', 'WND_3', 'WND_4', 'CIG_0', 'CIG_1', 'VIS_0', 'VIS_1', 'VIS_2', 'VIS_3', 'TMP_0', 'TMP_1', 'DEW_0', 'DEW_1', 'SLP_0', 'SLP_1', 'GA1_0', 'GA1_1', 'GA1_2', 'GA1_3', 'GA1_4', 'GA1_5']
group_col = ['DATE_JOIN_HH']
first_run_flag = 0

for g, gval in enumerate(group_col):
  for i, cval in enumerate(sum_var):
    query = """
    select 
       STATION
      ,"""+gval+""" +  INTERVAL -3 hours as DATE_JOIN_2H
      ,avg(""" +cval+""") as feature_name
      ,count(*) OBS_CNT
      ,'""" + cval + """' as Col_Name
      ,'""" + gval + """' as Sum_Name
    FROM df_weather_proc
    WHERE """ + cval + """ NOT IN (999,9999,99999,999999)
    GROUP BY STATION 
            ,"""+gval+""" +  INTERVAL -3 hours
    """
    result = sqlContext.sql(query)
    if first_run_flag == 0:
      final_result = result
      first_run_flag = 1
    else:    
      final_result = final_result.union(result)
final_result.createOrReplaceTempView("final_result")

# COMMAND ----------

# MAGIC %md ### Weather data: Part 4
# MAGIC 
# MAGIC Final step in the weather data ETL creates the table that will be used to join to airlines data. 

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
      
df_weather_summary = sqlContext.sql(SQL_TOP + SQL_FROM + SQL_JOIN)

# COMMAND ----------

# Setup the view so we can insert the data into DB
df_weather_proc = df_weather_summary
df_weather_proc.createOrReplaceTempView("df_weather_proc")

# save processed weather data. Run when finish with transformations.
dbutils.fs.rm(path + "/processed_data/" + "weather_processed.parquet", recurse =True)
df_weather_proc.write.format("parquet").save(path + "/processed_data/" + "weather_processed.parquet")

display(dbutils.fs.ls(path + "/processed_data/") )

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS df_weather_processed_sql; CREATE TABLE df_weather_processed_sql AS SELECT * FROM df_weather_proc; 

# COMMAND ----------

# MAGIC %md ## Airports latitude and longitude
# MAGIC 
# MAGIC In order to select the weather station closest to an airport we need to know latitude and longitude of the airports. For this we use airport information from https://data.humdata.org/dataset/ourairports-usa. 
# MAGIC 
# MAGIC We remove records of closed airports and records with null values in airports id. Nine airports are not available in this dataset (airports located in US territories) and are added to the dataframe. 
# MAGIC 
# MAGIC We create a dataframe containing airport id, latitude, longitude and region.

# COMMAND ----------

# File location and type

# Load data downloaded from https://data.humdata.org/dataset/ourairports-usa
file_location = "/FileStore/shared_uploads/amitk@berkeley.edu/us_airports.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","
  
# The applied options are for CSV files. For other file types, these will be ignored.
df_airports_location = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# only keep airports where the iata code is not null and airport is not closed
df_airports_location = df_airports_location.where((col('iata_code').isNotNull()) & (col('type') != 'closed'))

# print number of records
print(f'Number of airport locations loaded: {df_airports_location.count()}')

# select relevant columns 
airports_columns = ['iata_code', 'latitude_deg', 'longitude_deg', 'local_region']

# keep only relevant columns 
df_airports_location = df_airports_location.select(*airports_columns)

# print number of columns
print(f'Number of columns in airports dataset: {len(df_airports_location.columns)}')

# Add manually rows for nine airports corresponding mostly to US territories that are not in the uploaded csv
# Not in the cvs file: PSE, PPG, SPN, SJU, STX, GUM, BQN, ISN, STT
extra_airports = spark.createDataFrame([('PSE',18.0106,-66.5632,'PR'), ('PPG',-14.3290,-170.7132,'AS'), ('SPN',15.1197, 145.7283,'GU'),\
                                        ('SJU',18.4373, -66.0041,'PR'), ('STX',17.6995, -64.7975,'VI'), ('GUM',13.4853, 144.8008,'GU'), \
                                        ('BQN',18.4954, -67.1356,'PR'), ('ISN',48.1776, -103.6376,'ND'), ('STT',18.3361, -64.9723,'VI')
                                       ], ['iata_code', 'latitude_deg', 'longitude_deg', 'local_region'])

df_airports_location = df_airports_location.union(extra_airports)

# print number of records
print(f'Number of airport locations after adding data: {df_airports_location.count()}')


# COMMAND ----------

# MAGIC %md ## Stations data
# MAGIC In this section, we create a dataframe that contains the location (latitude and longitude) of the weather stations. We later use this data to find the station closest to an airport.

# COMMAND ----------

# load stations data 
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

# select relevant columns
stations_columns = ['station_id', 'lat', 'lon']

# keep only relevant columns and drop duplicates
df_stations = df_stations.select(*stations_columns).dropDuplicates()

# print number of records
print(f'Number of stations loaded: {df_stations.count()}')

# print number of columns
print(f'Number of columns in stations dataset: {len(df_stations.columns)}')

# COMMAND ----------

# MAGIC %md ## Closest weather station to each airport and time zone
# MAGIC 
# MAGIC Now that we have the location information of airports and weather stations, we can find the weather station that is closest to each airport based on latitude and longitude. 
# MAGIC 
# MAGIC To do this, we cross join the airports' location and stations' location dataframes, and then we calculate the distance between airports and stations. We select the station with minimum distance to an airport. To calculate the distance, we use a UDF (see User defined functions section above). 
# MAGIC 
# MAGIC Additionally, we calculate the time zone for all airports and stations. We use this information later to convert all timestamps to UTC. To calculate the time zone, we also use a UDF.

# COMMAND ----------

# Create a cross join between the airports location and stations location
df_distance = df_airports_location.crossJoin(broadcast(df_stations))

# Calculate distance between aiports and stations and store in new column
df_distance = df_distance.withColumn("haversine", haversine(col("latitude_deg"), col("longitude_deg"),col("lat"),col("lon")))

# register temp table to use sql to find the station with minimum distance to airport
df_distance.registerTempTable('df_distance_sql')

# create a dataframe 
# Note: there are 10 airport codes with 2 closest stations. The stations are on the same lat,long but different id. These airports don't exist in our
# airlines dataframe, so we can ignore this.
df_closest_station = spark.sql("select * from df_distance_sql where (iata_code,haversine ) in (select iata_code, min(haversine) from df_distance_sql group by iata_code) order by iata_code")

# rename columns 
df_closest_station = df_closest_station.withColumnRenamed('latitude_deg','airp_latitude') \
                                       .withColumnRenamed('longitude_deg','airp_longitude') \
                                       .withColumnRenamed('lat','station_latitude') \
                                       .withColumnRenamed('lon','station_longitude')

# get timezone for both airport and station
df_closest_station = df_closest_station.withColumn('airp_timezone', timezone(col('airp_latitude'), col('airp_longitude')))
df_closest_station = df_closest_station.withColumn('station_timezone', timezone(col('station_latitude'), col('station_longitude')))

# add a boolean column: True- airport and station are on the same timezone, False - otherwise
df_closest_station = df_closest_station.withColumn('same_timezone', col('airp_timezone') == col('station_timezone') )


# COMMAND ----------

# MAGIC %md ## Join the stations and time zone information with the airlines dataframe
# MAGIC In this section, we join the dataframe containing information about the closest stations to airports with the airlines dataframe. To do this join we consider origin and destination airports, in order to find the closest weather station for the origin and destination airport for each record in the airline's dataset.

# COMMAND ----------

# First: Join based on ORIGIN airport
# Left join the airlines dataframe with the dataframe that has the information of the closest station and timezone
df_main = df_airlines.join(broadcast(df_closest_station), df_airlines.ORIGIN == df_closest_station.iata_code, 'left')

# select columns to keep 
columns_main = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', \
                'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', \
                'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', \
                'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', \
                'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', \
                'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', \
                'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', \
                'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_datetime_scheduled', 'dep_datetime_actual', 'arriv_datetime_scheduled', \
                'arriv_datetime_actual', 'station_id', 'airp_timezone', 'station_timezone']

# keep columns of interest
df_main = df_main.select(*columns_main)

# rename columns added
df_main = df_main.withColumnRenamed('station_id','station_origin') \
                 .withColumnRenamed('airp_timezone','airp_origin_timezone') \
                 .withColumnRenamed('station_timezone','station_origin_timezone') 

# Second: Join based on DESTINATION airport
# Left join 
df_main2 = df_main.join(broadcast(df_closest_station), df_main.DEST == df_closest_station.iata_code, 'left')

# select columns to keep
columns_main2 = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', \
                 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', \
                 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', \
                 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', \
                 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', \
                 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', \
                 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', \
                 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_datetime_scheduled', 'dep_datetime_actual', 'arriv_datetime_scheduled', \
                 'arriv_datetime_actual', 'station_origin', 'airp_origin_timezone', 'station_origin_timezone', 'station_id', \
                  'airp_timezone', 'station_timezone']

# keep columns of interest
df_main2 = df_main2.select(*columns_main2)

# rename columns added
df_main2 = df_main2.withColumnRenamed('station_id','station_dest') \
                   .withColumnRenamed('airp_timezone','airp_dest_timezone') \
                   .withColumnRenamed('station_timezone','station_dest_timezone') 

# include timezone in the date/times of the flight
df_main2 = df_main2.withColumn('dep_datetime_scheduled_utc', to_utc_timestamp(col('dep_datetime_scheduled'), col('airp_origin_timezone'))) \
                   .withColumn('dep_datetime_actual_utc', to_utc_timestamp(col('dep_datetime_actual'), col('airp_origin_timezone'))) \
                   .withColumn('arriv_datetime_scheduled_utc', to_utc_timestamp(col('arriv_datetime_scheduled'), col('airp_dest_timezone'))) \
                   .withColumn('arriv_datetime_actual_utc', to_utc_timestamp(col('arriv_datetime_actual'), col('airp_dest_timezone'))) \
     



# COMMAND ----------

# save processed flights data
# Commented out to avoid saving again (takes time)- It is already in the folder. Run again if other transformations were applied to the data.

dbutils.fs.rm(path + "/processed_data/" + "airlines_processed.parquet", recurse =True)
df_main2.write.format("parquet").save(path + "/processed_data/" + "airlines_processed.parquet")

display(dbutils.fs.ls(path + "/processed_data/") )
