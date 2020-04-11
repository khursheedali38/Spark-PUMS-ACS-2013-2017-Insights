# Spark-PUMS-ACS-2013-2017-Insights
ACS PUMS 2013-2017 Knowledge Discovery on Spark Architecture. This is a part of AMOD 5410 course project. The problem to solve is to carry predictive analytics on the census data fo over 10M records. While the emphasis is not on getting the best accuracy, its on leveraging the spark architecture and core API using python (pyspark) to understand different constructs in spark and most importantly the working via the DAG's.

Used spark standlone cluster locally with a system 4 cores and 16GB RAM, running centOS.

### Prerequisites

What things you need to install the software and how to install them

+ Download [spark standalone](https://www.apache.org/dyn/closer.lua/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz)
```
wget 'https://www.apache.org/dyn/closer.lua/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz'
```
+ Untar the compiled spark version
```
tar xzvf spark-2.4.5-bin-hadoop2.7.tgz
```

### Installing

Install the packagaes and modules to work on which don't come along with pyspark

+ For python modules - pip
```
sudo yum install python-pip
```

+ For visualizations

```
pip install python-dist-explore
```

## Running

```
python <fileName.py>
```


## Built With

* [Apache Spark](https://spark.apache.org) - Large scale data processing 
* [Microsoft Azure](https://azure.microsoft.com/en-ca/) - Virtual Machine 


## Authors

* **Mohammed Khursheed Ali Khan**


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Spark Documentation
* Medium 
* TowardsDataScience
