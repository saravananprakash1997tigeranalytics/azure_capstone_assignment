# Databricks notebook source
#Mounting the Azure Storage account from Databricks Notebook
dbutils.fs.mount(source = "wasbs://prakashassignmentcontainer@prakashassignmentaccount.blob.core.windows.net",
                 mount_point = "/mnt/prakash_mounted_databricks",
                 extra_configs = {"fs.azure.account.key.prakashassignmentaccount.blob.core.windows.net" : "X3aTAwL1d5ah2jiEGqKHKA332XS6zv6K5g3jndokvMfYL68hHFibApSVExJkbkFbMkzW2ext8cnC+AStN6yVZw=="})

# COMMAND ----------

#Fetching or Reading the file from the blob storage
df = spark.read.csv("/mnt/prakash_mounted_databricks/Assignment_Folder_1/water_potability.csv", inferSchema=True, header = True)

# COMMAND ----------

df.display()

# COMMAND ----------

#Converting the Spark Dataframe to pandas Dataframe
pandas_df = df.select("*").toPandas()

# COMMAND ----------

pandas_df.isna().sum()

# COMMAND ----------

#Treating the Outliers in each of the columns in the dataset
for i in pandas_df.columns:
    q3=pandas_df[i].quantile(0.75)
    q1=pandas_df[i].quantile(0.25)
    IQR=q3-q1
    UL=q3+1.5*IQR
    LL=q1-1.5*IQR
    pandas_df.loc[pandas_df[i]>UL,i]=UL
    pandas_df.loc[pandas_df[i]<LL,i]=LL

# COMMAND ----------

#Replacing the null values of each independent variable with mean 
pandas_df =pandas_df.fillna(pandas_df.mean(axis=0))

# COMMAND ----------

pandas_df.isna().sum()

# COMMAND ----------

pandas_df

# COMMAND ----------

#Since it is a classification dataset , we are balancing the dataset with respect to the target column
pandas_df["Potability"].value_counts()

# COMMAND ----------

X = pandas_df.iloc[:,:-1].values
Y = pandas_df.iloc[:,-1].values

# COMMAND ----------

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros=RandomOverSampler(random_state=0)
X_resampled, y_resampled=ros.fit_resample(X,Y)
print(sorted(Counter(y_resampled).items()), y_resampled.shape)

# COMMAND ----------

preprocessed_data_output = pandas_df

# COMMAND ----------

#Converting the Dataframe back to Spark Dataframe
from pyspark.sql import SparkSession
#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()
#Create PySpark DataFrame from Pandas
sparkDF=spark.createDataFrame(preprocessed_data_output) 
sparkDF.printSchema()
sparkDF.show()


# COMMAND ----------

sparkDF.write.mode("overwrite").option("header", "true").csv(f"dbfs:/mnt/prakash_mounted_databricks/Azure_DF_Assignment_Folder/data_preprocessing_output.csv")

# COMMAND ----------

