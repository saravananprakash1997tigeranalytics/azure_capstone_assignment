# Databricks notebook source
df = spark.read.csv("/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/data_preprocessing_output.csv", inferSchema=True, header = True)

# COMMAND ----------

df.display()

# COMMAND ----------

df.columns

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
aasembler = VectorAssembler(inputCols = df.columns[:-1], outputCol= 'features')

# COMMAND ----------

output_df = aasembler.transform(df)

# COMMAND ----------

output_df.select("features", "Potability").show()

# COMMAND ----------

model_df = output_df.select("features","Potability")

# COMMAND ----------

training_df, test_df = model_df.randomSplit([0.75,0.25])

# COMMAND ----------

print(training_df.count(), test_df.count())

# COMMAND ----------

test_df.show(100)

# COMMAND ----------

#Saving the test Dataset

# from pyspark.sql.functions import col
# test_df = test_df.withColumn('features', col('features').cast('string'))

test_df.write.mode("overwrite").save(f"dbfs:/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/test_dataset_file")

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier,DecisionTreeClassificationModel
dt_classifier_model = DecisionTreeClassifier(labelCol = "Potability").fit(training_df)
try :
    dt_classifier_model.save("/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/deciontreeclassifiermodel")
except :
    #Overwriting the model file
    dbutils.fs.rm("/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/deciontreeclassifiermodel",recurse=True)
    dt_classifier_model.save("/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/deciontreeclassifiermodel")

# COMMAND ----------

