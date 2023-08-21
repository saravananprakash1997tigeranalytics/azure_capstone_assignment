# Databricks notebook source
from pyspark.ml.classification import DecisionTreeClassifier,DecisionTreeClassificationModel
saved_model = DecisionTreeClassificationModel.load("/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/deciontreeclassifiermodel")

# COMMAND ----------

loaded_test_df = spark.read.load("/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/test_dataset_file")

# COMMAND ----------

loaded_test_df.display()

# COMMAND ----------

predictions = saved_model.transform(loaded_test_df)

# COMMAND ----------

predictions.show()

# COMMAND ----------

loaded_test_df.write.mode("overwrite").save(f"dbfs:/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/predictions_dataset")

# COMMAND ----------

dbutils.fs.unmount("/mnt/prakash_mounted_databricks")

# COMMAND ----------

