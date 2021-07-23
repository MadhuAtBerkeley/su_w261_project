# Databricks notebook source
from pyspark.sql.functions import col, max
blob_container = "w261" # The name of your container created in https://portal.azure.com
storage_account = "w261" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

dbutils.fs.mount(
  source = blob_url,
  mount_point = "/mnt/Azure",
extra_configs = {"fs.azure.account.key.w261.blob.core.windows.net":dbutils.secrets.get(scope = secret_scope, key = secret_key)})

# COMMAND ----------

#List files
dbutils.fs.ls("/mnt/Azure")

# COMMAND ----------

df = spark.read.text("/mnt/Azure/README.md")
df.head(10)

# COMMAND ----------

display(dbutils.fs.ls(f"{mount_path}"))

# COMMAND ----------



# COMMAND ----------

#List files
dbutils.fs.ls("/mnt/mids-w261/HW5")

# COMMAND ----------

#List files
dbutils.fs.ls("/mnt/mids-w261/datasets_final_project")

# COMMAND ----------

#List files
dbutils.fs.ls("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m")

# COMMAND ----------

# COPY FILE TO BUCKET
dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/part-00000-e7b7f524-7be7-4ec3-afce-c31684af3aca-c000.snappy.parquet", "/mnt/Azure/part-00000-e7b7f524-7be7-4ec3-afce-c31684af3aca-c000.snappy.parquet")

# COMMAND ----------

# COPY FOLDER TO BUCKET
dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m", "/mnt/Azure/parquet_airlines_data_3m",True)

# COMMAND ----------

# COPY FOLDER TO BUCKET
dbutils.fs.cp("/mnt/mids-w261/datasets_final_project", "/mnt/Azure/datasets_final_project",True)

# COMMAND ----------

# COPY FOLDER TO BUCKET
dbutils.fs.cp("/mnt/mids-w261/HW5", "/mnt/Azure/HW5",True)

# COMMAND ----------

dbutils.fs.unmount("/mnt/Azure")
