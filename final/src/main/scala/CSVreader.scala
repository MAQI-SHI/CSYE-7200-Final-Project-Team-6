
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorIndexer, StringIndexer}

import org.apache.spark.sql.SparkSession
class CSVreader {

}
object Prediction{
  def main(args: Array[String]): Unit = {
    //create spark object
    val spark = SparkSession.builder()
          .appName("RandomForest")
          .master("local[2]")
          .getOrCreate()
    //get data from csv file
    val healthData = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/healthcare-dataset-stroke-data.csv")
    //healthData.printSchema()
    //healthData.show()
    //Identify the identity column and index column of the entire data set
    val labelIndexer = new StringIndexer()
           .setInputCol("label")
           .setOutputCol("indexedLabel")
           .fit(healthData)
  }
}
