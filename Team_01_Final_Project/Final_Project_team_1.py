# Databricks notebook source
# MAGIC %md # Flight departure delay predictions
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
# MAGIC Many people have worked in understanding the problem of flights delays. For instance, in 2019 Navoneel Chakrabarty* developed a model to predict arrival delay for American Airlines flights in US, that achieved an accuracy of 85.73%, which is considered a very good accuracy for this type of problems.
# MAGIC 
# MAGIC The implementation of a model to predict flight delays would have a great impact on airline operations. Airlines would be able to minimize the impact of such delays by making changes on passenger itineraries, flight routes, crew assignments, aircraft schedules and maintenance, etc. 
# MAGIC 
# MAGIC The main purpose of this study is to create a model to predict **departure delay** for flights in the US, where a delay is defined as 15-minute delay or more. We use a subset of the of the flight's on-time performance data provided by the United States Bureau of Transportation Statistics. The data comprises data of flights departing from all major US airports for the 2015-2019 timeframe. Additionally, we use weather information from the National Oceanic and Atmospheric Administration repository.
# MAGIC 
# MAGIC The output variable in our model is a binary variable, where 1 represent flights that experienced departure delay and 0 represent flights with on-time departure. 
# MAGIC 
# MAGIC In order to measure the model performance, we use precision and recall. We are not using accuracy, because accuracy does not provide information about the quality of a classifier in predicting a specific class (delayed/on-time). We want to be able to assess how good is our classifier with respect to a specific class. Additionally, accuracy is not a good measure when classes are unbalanced.
# MAGIC 
# MAGIC We define delay precision and recall as:
# MAGIC $$Prec_{delay} = \frac{TP}{TP + FP}$$
# MAGIC $$Recall_{delay} = \frac{TP}{TP + FN}$$
# MAGIC 
# MAGIC 
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From the project instructions:
# MAGIC >
# MAGIC > You should refine the question formulation based on the general task description you’ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etc.. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics.
# MAGIC >
# MAGIC > From rubric:
# MAGIC >
# MAGIC > - Data set contents and context are clearly introduced.
# MAGIC > - Clearly articulated question that is appropriate to the both the data and the algorithm and takes limitations of the data and/or algorithm into account.
# MAGIC > - Evaluation metrics are discussed - How will you know if you have a good model?
# MAGIC > -Specify baselines, state-of-the-art literature or results for comparison

# COMMAND ----------

# MAGIC %md ## Exploratory data analysis
# MAGIC 
# MAGIC In this section we explore the flight dataset and the weather dataset.
# MAGIC 
# MAGIC We use the smaller flight dataset (either the three or six months of flight data, **we need to decide**), but some EDA needs to be done on the complete flight dataset.
# MAGIC 
# MAGIC - 1 Quarter dataset:  includes flights departing from two major US airports (ORD (Chicago O’Hare) and ATL (Atlanta) for the first quarter of 2015 (that is about 160k flights) .
# MAGIC - 2 Quarters dataset:  includes flights from the same two airports for the first six months of 2015.
# MAGIC 
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From project description:
# MAGIC >
# MAGIC > Determine a handful of relevant EDA tasks that will help you make decisions about how you implement the algorithm to be scalable. Discuss any challenges that you anticipate based on the EDA you perform.
# MAGIC > 
# MAGIC > From rubric:
# MAGIC > - 3-4 EDA tasks.
# MAGIC > - EDA tasks are well chosen and well explained.
# MAGIC > - Code is scalable & well commented.
# MAGIC > - Written discussion connects the EDA to the algorithm and/or potential challenges.

# COMMAND ----------

# MAGIC %md ### Package Imports

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md ### Inspect and load data

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

# MAGIC %md ## Data preparation and feature engineering
# MAGIC 
# MAGIC In this section we preprocess the data to look for inconsistencies and missing values. We also discard not important information, encode categorical variables, create new features and normalize the data (**if needed**). At the end we create a joined dataset including data from the flight and weather dataset.
# MAGIC 
# MAGIC We use the smaller flight dataset for testing new features, but the features are created in the complete flight dataset. 
# MAGIC 
# MAGIC 
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From project instructions:
# MAGIC >
# MAGIC > Apply relevant feature transformations, dimensionality reduction if needed, interaction terms, treatment of categorical variables, etc.. Justify your choices.
# MAGIC >
# MAGIC > From rubric:
# MAGIC > - Clearly explain each column.
# MAGIC > - Justify missing value treatment
# MAGIC > - Address the feature distributions, and justify any transformations

# COMMAND ----------

# MAGIC %md ### Missing Values
# MAGIC 
# MAGIC Deal with missing values on the flight dataset and the weather dataset.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Remove not relevant weather observations
# MAGIC 
# MAGIC In this section we remove from the weather table all observations that are not related to airports location.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Categorical variables
# MAGIC 
# MAGIC In this section we encode the categorical variables.
# MAGIC 
# MAGIC To deal with categorical variables I propose to use binary encoding to avoid the problem of too many dimensions of the one-hot-encoder or dummy variables approach. A good resource explaining different encoding methods here (https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Create new features
# MAGIC 
# MAGIC In this section we create new features.
# MAGIC 
# MAGIC Some ideas for features:
# MAGIC - Feature that represent if a date is a holiday. Not sure if such feature exists already in the dataset, if not this is a good reference for Holiday dates (https://www.transtats.bts.gov/holidaydelay.asp)
# MAGIC - Feature to represent seasons (spring, summer, fall, winter)
# MAGIC - PageRank (class suggestion)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Join tables
# MAGIC In this section we join the flight table and the weather table. For each record in the flight table (each flight) we include weather observations at origin airport and at destination airport. Since the purpose of this project is to predict if a flight departure will be delayed two hours ahead of departure, the weather data that we consider does not go beyond 2 hours before the scheduled departure of the flight. Hence, we consider hourly weather observations at both origin and destination airport that go from t-8h to t-2h, where t is the scheduled departure time.
# MAGIC 
# MAGIC The join operation is split in two steps: the first join step combines flight information with weather observations at origin airport, and the second join step com- bines the output of the first step with the weather observations at destination airport.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Normalization
# MAGIC 
# MAGIC In this section we normalize the features (if needed).

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Address imbalanced data
# MAGIC 
# MAGIC In this section we explore a method to address the problem of the data imbalance.
# MAGIC 
# MAGIC We could create two base models, one using the dataset where we address the problem of the imbalance and another without addressing the problem to see if it helps or not.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Split dataset into train, development and test datasets
# MAGIC We need to be mindful of the sampling method given that the data is timeseries.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Modeling
# MAGIC 
# MAGIC In this section we explore a base models.
# MAGIC 
# MAGIC Base model: I would propose something simple such as Logistic Regression and another one.
# MAGIC 
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From project instructions:
# MAGIC >
# MAGIC >Apply 2 to 3 algorithms to the training set, and discuss expectations, trade-offs, and results. These will serve as your baselines - do not spend too much time fine tuning these. You will want to use this process to select a final algorithm which you will spend your efforts on fine tuning.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Algorithm Implementation
# MAGIC 
# MAGIC I suggest we do something relatively simple like logistic regression here.
# MAGIC 
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From project instructions:
# MAGIC >
# MAGIC > Create your own toy example that matches the dataset provided and use this toy example to explain the math behind the algorithm that you will perform. Apply your algorithm to the training dataset and evaluate your results on the test set. 
# MAGIC >
# MAGIC > From rubric:
# MAGIC > - Math clearly explained without overly technical language
# MAGIC > - Toy example is appropriate to the algorithm & dataset.
# MAGIC > - Toy example calculations are clearly presented

# COMMAND ----------



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



# COMMAND ----------

# MAGIC %md ## Application of Course Concepts
# MAGIC > **Reference from instructions and rubric**
# MAGIC >
# MAGIC > From project instructions:
# MAGIC >
# MAGIC > Pick 3-5 key course concepts and discuss how your work on this assignment illustrates an understanding of these concepts.
# MAGIC >
# MAGIC > From rubric:
# MAGIC > - Correctly identifies 3-5 course concepts that are relevant to this algorithm.
# MAGIC > - Discussion demonstrates understanding of the chosen concepts and addresses how the algorithm performance/scalability/ practicality would have been different given a different implementation.
# MAGIC > - (½ - 1 page; ~3 paragraphs)

# COMMAND ----------


