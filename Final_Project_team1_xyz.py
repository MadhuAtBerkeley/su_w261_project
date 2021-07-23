# Databricks notebook source
# MAGIC %md # Flight departure delay predictions
# MAGIC 
# MAGIC Team members:
# MAGIC - Isabel Garcia Pietri
# MAGIC - Madhu Hegde
# MAGIC - Amit Karandikar
# MAGIC - Piotr Parkitny

# COMMAND ----------

# MAGIC %md ### Package Imports and configuration

# COMMAND ----------

# install packages
%pip install timezonefinder

# COMMAND ----------

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType

import numpy as np
import math
from timezonefinder import TimezoneFinder


# COMMAND ----------

# spark configuration
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md ### Data ingestion

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# MAGIC %md #### Airlines data
# MAGIC 
# MAGIC The airlines dataset has 63,493,682 total records. This dataset includes 979,894 flights cancelled for many reasons: carrier caused, weather, national aviation system and security. These flights could be considered as extreme delays, where passengers are relocated to other flights, however for the purpose of this study we decided to filter out flights that are cancelled. 
# MAGIC 
# MAGIC Diverted flights are kept as we are interested in predicting delays at departure, independently if the flight arrives to the destination airport or to another airport due to a diversion.

# COMMAND ----------

# Load airlines data
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")

# filter out cancelled flights
df_airlines = df_airlines.where(col('CANCELLED') != 1)

# print number of records
print(f'Number of flight records loaded: {df_airlines.count()}')

# Select columns to keep. Remove cancellation related columns, airport diversion related columns
airlines_columns = ['YEAR','QUARTER','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','FL_DATE','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM',\
                    'OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN_CITY_MARKET_ID','ORIGIN','ORIGIN_CITY_NAME','ORIGIN_STATE_ABR',\
                    'ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID','DEST','DEST_CITY_NAME',\
                    'DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW','DEP_DEL15',\
                    'DEP_DELAY_GROUP','DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', \
                    'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME','DISTANCE',
                    'DISTANCE_GROUP','CARRIER_DELAY', 'WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']

# keep relevant columns 
df_airlines = df_airlines.select(*airlines_columns)

# print number of columns
print(f'Number of columns in airlines dataset: {len(df_airlines.columns)}')

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

# COMMAND ----------

# MAGIC %md #### Weather data

# COMMAND ----------

# load weather dataset
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")

# keep only the data coming from the stations in the US
df_weather = df_weather.where(col('NAME').endswith('US'))

# print number of records
print(f'Number of weather records loaded: {df_weather.count()}')

# select columns to keep
weather_columns = ['STATION', 'DATE', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND', \
                   'CIG', 'VIS', 'TMP', 'DEW', 'SLP']

# keep relevant columns
df_weather = df_weather.select(*weather_columns)

# print number of columns
print(f'Number of columns in weather dataset: {len(df_weather.columns)}')

# unpack columns containing comma separated values- MISSING


# COMMAND ----------

# MAGIC %md #### Airports latitute and longitude

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

# MAGIC %md #### Stations data

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

# MAGIC %md #### Closest weather station to each airport and timezone

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

# MAGIC %md #### Join the stations and timezone information with the airlines dataframe

# COMMAND ----------

# First: Join based on ORIGIN airport
# Left join the airlines dataframe with the dataframe that has the information of the closest station and timezone
df_main = df_airlines.join(broadcast(df_closest_station), df_airlines.ORIGIN == df_closest_station.iata_code, 'left')

# select columns to keep 
columns_main = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER',\
                'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', \
                'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', \
                'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', \
                'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', \
                'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',\
                'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_time_scheduled', \
                'dep_datetime_scheduled', 'dep_time_actual', 'dep_datetime_actual', 'station_id', 'airp_timezone', 'station_timezone']

# keep columns and change name of added columns
df_main = df_main.select(*columns_main)

# rename columns added
df_main = df_main.withColumnRenamed('station_id','station_origin') \
                 .withColumnRenamed('airp_timezone','airp_origin_timezone') \
                 .withColumnRenamed('station_timezone','station_origin_timezone') 

# Second: Join based on DESTINATION airport
# Left join 
df_main2 = df_main.join(broadcast(df_closest_station), df_main.DEST == df_closest_station.iata_code, 'left')

# select columns to keep
columns_main2 = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER',\
                'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', \
                'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', \
                'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', \
                'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', \
                'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME','DISTANCE', \
                'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'dep_time_scheduled', \
                'dep_datetime_scheduled', 'dep_time_actual', 'dep_datetime_actual', 'station_origin', 'airp_origin_timezone', 'station_origin_timezone', \
                'station_id', 'airp_timezone', 'station_timezone']

# keep columns and change name of added columns
df_main2 = df_main2.select(*columns_main2)

# rename columns added
df_main2 = df_main2.withColumnRenamed('station_id','station_dest') \
                   .withColumnRenamed('airp_timezone','airp_dest_timezone') \
                   .withColumnRenamed('station_timezone','station_dest_timezone') 

# include timezone in the date/times of the flight
df_main2 = df_main2.withColumn('dep_datetime_scheduled_utc', to_utc_timestamp(col('dep_datetime_scheduled'), col('airp_origin_timezone'))) \
                   .withColumn('dep_time_actual_utc', to_utc_timestamp(col('dep_time_actual'), col('airp_origin_timezone'))) 


# COMMAND ----------

display(df_main2)

# COMMAND ----------

# MAGIC %md #### register sql tables for direct sql queries

# COMMAND ----------

# register temp tables for sql queries
df_airlines.registerTempTable('df_airlines_sql')
df_weather.registerTempTable('df_weather_sql')
df_closest_station.registerTempTable('df_closest_station_sql')
df_main2.registertempTable('df_main2_sql')
