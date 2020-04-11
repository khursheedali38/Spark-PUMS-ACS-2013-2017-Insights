# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as f
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from matplotlib import pyplot as plt
from pyspark_dist_explore import hist, Histogram, distplot
import seaborn as sns
import pandas as pd

get_ipython().magic(u'matplotlib inline')


# %%
spark = SparkSession.builder.master(
    "local[*]").appName("ACS PUMS 2013-2017 Data Analysis").getOrCreate()


# %%
# loading the dataset as DataFrame API, one at a time cleaning and removing it unitll next load
dataA = spark.read.csv('data/csv_pus/csv_pus/psam_pusa.csv',
                       header=True, inferSchema=True)
dataB = spark.read.csv('data/csv_pus/csv_pus/psam_pusb.csv',
                       header=True, inferSchema=True)
dataC = spark.read.csv('data/csv_pus/csv_pus/psam_pusc.csv',
                       header=True, inferSchema=True)
dataD = spark.read.csv('data/csv_pus/csv_pus/psam_pusd.csv',
                       header=True, inferSchema=True)


# %%
# selecting the columns we need - tranformation
cols = ['REGION', 'ST', 'AGEP', 'CIT', 'CITWP', 'COW', 'DDRS', 'DEAR', 'DEYE', 'DREM', 'ENG', 'FER', 'GCL', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6',
        'HINS7', 'JWMNP', 'JWRIP', 'JWTR', 'LANX', 'MAR', 'RELP', 'SCH', 'SCHG', 'SCHL', 'SEX', 'FOD1P', 'FOD2P', 'INDP', 'JWAP', 'JWDP', 'LANP', 'POVPIP', 'POWSP']

dataA = dataA.select(*cols)
dataB = dataB.select(*cols)
dataC = dataC.select(*cols)
dataD = dataD.select(*cols)


# %%
# wrangle data dictionary to perform mapping
# map categorical values to data-dictionary
# combine the 4 data files
# perform the anlaysis such as one-hot encoding etc
# clustering
# mlib
# graphX based on Relation attribute


# %%
# load data dictionary
# join woth data Dictionary only during plotting as legend
dataDict = spark.read.csv(
    'data/csv_pus/csv_pus/PUMS_Data_Dictionary_2013-2017.csv', header=True, inferSchema=True)


# %%
dataDict = dataDict.dropna(how='any')
dataDict = dataDict.select('RT', 'Record Type', '_c6')


# %%
dataDict = dataDict.filter(f.col('RT').isin(cols))


# %%
dataDict = dataDict.withColumnRenamed('RT', 'colName')
dataDict = dataDict.withColumnRenamed('Record Type', 'code')
dataDict = dataDict.withColumnRenamed('_c6', 'abbrev')
dataDict.show()


# %%
# get the session details
# spark.sparkContext.getConf().getAll()
dataDict.filter(dataDict.colName == 'SEX').show()


# %%
spark.sparkContext.uiWebUrl


# %%
# merge all the dataframes into one dataframe
df = dataA.union(dataB)
df = df.union(dataC)
df = df.union(dataD)


# %%
# clearing the dataframes to free up the memory
dataA.unpersist(True)
dataB.unpersist(True)
dataC.unpersist(True)
dataD.unpersist(True)


# %%
# finding the insights on data,
# the vision etc difficulties dmographics and mostly on high gage groups
# Create some selections on this data
filtered_by_gender_m = df.filter(f.col('SEX') == '1').select('INDP').join(dataDict, [
    dataDict.colName == 'INDP', dataDict.code.cast('int') == df.INDP], 'inner').select(f.col('abbrev').alias('INDUSTRY_m'))
filtered_by_gender_f = df.filter(f.col('SEX') == '2').select('INDP').join(dataDict, [
    dataDict.colName == 'INDP', dataDict.code.cast('int') == df.INDP], 'inner').select(f.col('abbrev').alias('INDUSTRY_f'))

filtered_by_age_50_plus = df.filter(f.col('AGEP') > 50).select('COW').join(dataDict, [
    dataDict.colName == 'COW', dataDict.code.cast('int') == df.COW], 'inner').select(f.col('abbrev').alias('ClassOfWorker_50_plus'))
filtered_by_age_50_minus = df.filter(f.col('AGEP') <= 50).select('COW').join(dataDict, [
    dataDict.colName == 'COW', dataDict.code.cast('int') == df.COW], 'inner').select(f.col('abbrev').alias('ClassOfWorker_50_minus'))


# %%
# aggregating the data for easier and memory efficient pandas plotting
vizM = filtered_by_gender_m.groupBy(
    'INDUSTRY_m').count().sort('count', ascending=False)
vizF = filtered_by_gender_f.groupBy(
    'INDUSTRY_f').count().sort('count', ascending=False)

viz50Minus = filtered_by_age_50_minus.groupBy(
    'ClassOfWorker_50_minus').count().sort('count', ascending=False)
viz50Plus = filtered_by_age_50_plus.groupBy(
    'ClassOfWorker_50_plus').count().sort('count', ascending=False)


# %%
# panadas plotting visualizations
# visualization issues with long labels, so use string indexer
C = vizM.toPandas()
D = vizF.toPandas()

pd.concat({
    'Industry M': C.set_index('INDUSTRY_m'), 'Industry F': D.set_index('INDUSTRY_f')
}, axis=1).plot.barh()


# %%
A = viz50Minus.toPandas()
B = viz50Plus.toPandas()

pd.concat({
    '50 Plus COW': B.set_index('ClassOfWorker_50_plus'), '50 Minus COW': A.set_index('ClassOfWorker_50_minus')
}, axis=1).plot.barh()


# %%
filtered_by_gender_m.groupBy('INDUSTRY_m').count().sort(
    'count', ascending=False).show()


# %%
filtered_by_gender_f.groupBy('INDUSTRY_f').count().sort(
    'count', ascending=False).show()


# %%
filtered_by_age_50_minus.groupBy('ClassOfWorker_50_minus').count().sort(
    'count', ascending=False).show()


# %%
filtered_by_age_50_plus.groupBy('ClassOfWorker_50_plus').count().sort(
    'count', ascending=False).show()


# %%
# determing the inDustry of work using Machine Learning
#features = [AGEP, SCH, SCHG, SCHL, SEX, FOD1P, FOD2P, COW]
#target = INDP

# AGEP > 17 takes care of COW and Schooling
# nit doing joimn with the dataDict as during one hot encoding the column names might have spaces and doesn't make sense, c an use join during plotting and graphing
dfX = df.filter((f.col('AGEP') > 17) & (f.col('INDP').isNotNull())).select(
    'AGEP', 'SCHL', 'SEX', 'FOD1P', 'FOD2P', 'COW', 'INDP')
dfX = dfX.fillna({'FOD1P': 0, 'FOD2P': 0})


# %%
dfX.show()


# %%
# onehotencoding of cateforical variablesa for classification
categCol = ['SCHL', 'SEX', 'FOD1P', 'FOD2P', 'COW']
numericCol = ['AGEP']
stages = []

# StringIndexing and oneHotEncoding of categorical features
for cols in categCol:
    stringIndexer = StringIndexer(inputCol=cols, outputCol=cols+'Index')
    encoder = OneHotEncoderEstimator(
        inputCols=[stringIndexer.getOutputCol()], outputCols=[cols + 'classVec'])
    stages += [stringIndexer, encoder]

# String indexing of label
labelStringIndexer = StringIndexer(inputCol='INDP', outputCol='label')
stages += [labelStringIndexer]

assemblerInput = [c + 'classVec' for c in categCol] + numericCol
assembler = VectorAssembler(inputCols=assemblerInput, outputCol="features")
stages += [assembler]

# %%
# creating a ML pipeline
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(dfX)
dfX = pipelineModel.transform(dfX)
dfX = dfX.select('label', 'features')


# %%
dfX.show()


# %%
# train test split
train, test = dfX.randomSplit([0.8, 0.2], seed=2020)
print("train count: " + str(train.count()))
print("test count: " + str(test.count()))


# %%
dctClf = DecisionTreeClassifier(
    featuresCol='features', labelCol='label', maxDepth=3)
dctModel = dctClf.fit(train)
predictions = dctModel.transform(test)
predictions.select('prediction', 'probability').show(10)


# %%
# peerformance metric, evaluation
evaluator = MulticlassClassificationEvaluator()
print("Area under Curve ROC: " +
      str(evaluator.evaluate(predictions, {evaluator.metricName: "f1"})))


# %%
# RandomForesrt classifier
rfClf = RandomForestClassifier(featuresCol='features', labelCol='label')
rfModel = rfClf.fit(train)
predictions = rfModel.transform(test)
predictions.select('prediction', 'probability').show(10)


# %%
evaluator = MulticlassClassificationEvaluator()
print("Accuracy: " + str(evaluator.evaluate(predictions,
                                            {evaluator.metricName: "accuracy"})))


# %%

