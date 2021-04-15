
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorIndexer, StringIndexer}

import org.apache.spark.sql.SparkSession
class CSVreader {

}
object Prediction{
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
          .appName("RandomForest")
          .master("local[2]")
          .getOrCreate()
    val healthdata = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/healthcare-dataset-stroke-data.csv")
    healthdata.printSchema()
    healthdata.show()

  }
}
