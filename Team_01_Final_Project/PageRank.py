# Databricks notebook source
# MAGIC %md # Flight departure delay predictions
# MAGIC ## PageRank and proxy categorical variables for airports
# MAGIC 
# MAGIC Team members:
# MAGIC - Isabel Garcia Pietri
# MAGIC - Madhu Hegde
# MAGIC - Amit Karandikar
# MAGIC - Piotr Parkitny

# COMMAND ----------

# MAGIC %md ## Package imports, directories and configuration

# COMMAND ----------

# install networkx for visualization (Spark does not have a native way to create a visualization of a graph)
%pip install networkx

# COMMAND ----------

# install geopandas (to aid visualization)
%pip install geopandas

# COMMAND ----------

# package imports
from pyspark.sql.functions import col, isnull, broadcast, udf, count, when, isnan, lpad, to_timestamp, concat, to_utc_timestamp, lit 
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import Bucketizer

from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType
from functools import reduce
from graphframes import *

import numpy as np
import math

import networkx as nx
import matplotlib.pyplot as plt
import geopandas


# COMMAND ----------

#data path to our directory 
path = 'dbfs:/mnt/Azure/'
display(dbutils.fs.ls(path+'/processed_data'))

# COMMAND ----------

# spark configuration
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md ## Functions
# MAGIC We create two functions to help create a visualization using the networkx package.

# COMMAND ----------

# function to create networkx graph from edges in graphx
def create_networkx(edge_list):
    g = nx.Graph()
    for row in edge_list.select('src','dst').take(8000):
        g.add_edge(row['src'],row['dst'])
    return g
  
# function to create a node position dictionary from dataframe
def pos_dict(position):
  # initialize empty dictionary
  pos = {}
  for row in position.select('id','longitude_deg','latitude_deg').take(400):
    pos[row['id']] = (row['longitude_deg'], row['latitude_deg'])
    
  return pos



# COMMAND ----------

# MAGIC %md ## PageRank
# MAGIC 
# MAGIC In this notebook we calculate the PageRank for the airports in the dataset. We first create a graph dataframe and then use it to calculate the PageRank.
# MAGIC 
# MAGIC The PageRank algorithm measures the importance of each node (in this case airport) within the graph, based on the number incoming relationships (incoming routes from other airports) and the importance of the corresponding source nodes (other airports). The assumption is that if an airport with many connections is delayed, this delay will likely propagate to other airports.
# MAGIC 
# MAGIC Using the nodes degrees in the graph, we also create a proxy categorical variable to represent the airports based on their size (number of incoming/outgoing connections).

# COMMAND ----------

# Load pre-processed airlines data

df_airlines = spark.read.parquet("/mnt/Azure/processed_data/airlines_processed.parquet")

print(f'{df_airlines.count():,} flight records loaded')


# COMMAND ----------

# extract unique origin and destination airports
origin_airports = df_airlines.select('ORIGIN').distinct()
dest_airports = df_airlines.select('DEST').distinct()
vertices = origin_airports.union(dest_airports).distinct()

# COMMAND ----------

# create vertex dataframe. A vertex DataFrame should contain a special column named "id" which specifies unique IDs for each vertex 
vertices = vertices.withColumnRenamed('ORIGIN', 'id') 

# COMMAND ----------

display(vertices)

# COMMAND ----------

# create an edge dataframe. 
# An edge DataFrame should contain two special columns: "src" (source vertex ID of edge) and "dst" (destination vertex ID of edge).
edges = df_airlines.select('ORIGIN', "DEST").drop_duplicates() \
                   .withColumnRenamed('ORIGIN', 'src') \
                   .withColumnRenamed('DEST', 'dst') 

# COMMAND ----------

display(edges)

# COMMAND ----------

edges.count()

# COMMAND ----------

# create a graph dataframe 
from graphframes import GraphFrame
g = GraphFrame(vertices, edges)
print(g)

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

# MAGIC %md Nodes with no edges going in: EFD, ENV, TKI

# COMMAND ----------

display(g.outDegrees)

# COMMAND ----------

# MAGIC %md Nodes with no edges going out: FNL. Need to consider teleporting when calculating the PageRank.

# COMMAND ----------

g_degree = g.degrees
display(g_degree)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 1
results = g.pageRank(resetProbability=0.15, maxIter=1)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 2
results = g.pageRank(resetProbability=0.15, maxIter=2)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 3
results = g.pageRank(resetProbability=0.15, maxIter=3)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 4
results = g.pageRank(resetProbability=0.15, maxIter=4)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 5
results = g.pageRank(resetProbability=0.15, maxIter=5)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 6
results = g.pageRank(resetProbability=0.15, maxIter=6)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 7
results = g.pageRank(resetProbability=0.15, maxIter=7)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 8
results = g.pageRank(resetProbability=0.15, maxIter=8)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 9
results = g.pageRank(resetProbability=0.15, maxIter=9)
display(results.vertices)

# COMMAND ----------

# Calculate pagerank. Number of iterations = 10
results = g.pageRank(resetProbability=0.15, maxIter=10)
display(results.vertices)

# COMMAND ----------

# MAGIC %md We stop at 10 iterations as the difference between iterations is low (and the processing time is high). Comparing the results of 9 iterations and 10 iterations: maximum percentage change in the PageRank values is 1.5%, and in average the percentage change in the PageRank values is 0.4%. 

# COMMAND ----------

# Save pagerank results to parquet

path = 'dbfs:/mnt/Azure/'

dbutils.fs.rm(path + "/processed_data/" + "pagerank.parquet", recurse =True)
results.vertices.write.parquet(path + "/processed_data/" + "pagerank.parquet")

display(dbutils.fs.ls(path + "/processed_data/") )

# g.vertices.write.parquet("hdfs://myLocation/vertices")

# COMMAND ----------

# MAGIC %md ## Proxy Categorical variables for airports
# MAGIC 
# MAGIC Using the nodes degrees in the graph, we create a proxy categorical variable to represent the airports based on their size (number of incoming/outgoing connections).

# COMMAND ----------

# Load previusly calculated PageRank

df_pagerank = spark.read.parquet("/mnt/Azure/processed_data/pagerank.parquet")

print(f'{df_pagerank.count():,} PageRank records loaded')

# COMMAND ----------

display(df_pagerank)

# COMMAND ----------

# rename id column in g_degree dataframe before merging
g_degree = g_degree.withColumnRenamed('id','id2')
display(g_degree)

# COMMAND ----------

# join PageRank data and degrees data
df_page_degree = df_pagerank.join(broadcast(g_degree), df_pagerank.id == g_degree.id2, 'left')

# list of columns to keep
columns = ['id', 'pagerank', 'degree']

# select columns to keep
df_page_degree = df_page_degree.select(*columns)

# COMMAND ----------

display(df_page_degree)

# COMMAND ----------

# group airports in buckets depending on the degree

bucketizer = Bucketizer(splits=[0,10,50,150,300,400],inputCol='degree', outputCol='airport_size_group')

df_buck = bucketizer.setHandleInvalid('keep').transform(df_page_degree)

# COMMAND ----------

display(df_buck)

# COMMAND ----------

# Save results to parquet- This file contains both the information of PageRank, degree and proxy categorical variable

path = 'dbfs:/mnt/Azure/'

dbutils.fs.rm(path + "/processed_data/" + "pagerank_degree.parquet", recurse =True)
df_buck.write.parquet(path + "/processed_data/" + "pagerank_degree.parquet")

display(dbutils.fs.ls(path + "/processed_data/") )


# COMMAND ----------

# MAGIC %md ## Graph visualization
# MAGIC In this section we create a visualization of the graph.

# COMMAND ----------

# MAGIC %md ### Get latitude and longitude
# MAGIC Use the same code of the main notebook.

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

display(df_airports_location)

# COMMAND ----------

# display graphframe edges
display(g.edges)

# COMMAND ----------

# create graph in networkx format using the helper function we created
graph = create_networkx(g.edges)

# COMMAND ----------

# create a dataframe with vertices (airports) and longitude and latitude
position = vertices.join(broadcast(df_airports_location), vertices.id == df_airports_location.iata_code, 'left')
position = position.select(['id', 'longitude_deg', 'latitude_deg'])
display(position)

# COMMAND ----------

# create node position (longitude, latitude) dictionary using the helper function we created
pos = pos_dict(position)

# COMMAND ----------

# add position attribute to the graph
nx.set_node_attributes(graph, pos, 'pos')

# COMMAND ----------

# eliminate nodes that are outside of US land to improve the visualization

# remove them from the position dictionary 
pos.pop('GUM')
pos.pop('SPN')
pos.pop('PPG')

# remove nodes in graph 
graph.remove_node('GUM')
graph.remove_node('SPN')
graph.remove_node('PPG')

# COMMAND ----------

# load world map information from geopandas
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# create graph visualization
fig, ax = plt.subplots(figsize=(20, 15))
nx.draw(graph, nx.get_node_attributes(graph, 'pos'), with_labels=False, node_size=50, node_color='blue', edge_color='grey' ,ax=ax)
world[world.continent =='North America'].plot(alpha=0.5, edgecolor=None, ax=ax)
limits=plt.axis('on') # turns on axis
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# COMMAND ----------


