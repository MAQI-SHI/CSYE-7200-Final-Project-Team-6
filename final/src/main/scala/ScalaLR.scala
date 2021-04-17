
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression

class ScalaLR{

}

object logic {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("logic")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val data = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/healthcare-dataset-stroke-data.csv")

    val indexedGender = new StringIndexer()
      .setInputCol("gender")
      .setOutputCol("indexedGender")
      .setHandleInvalid("keep")
      .fit(data)

    val indexedMarried= new StringIndexer()
      .setInputCol("ever_married")
      .setOutputCol("indexedMarried")
      .setHandleInvalid("keep")
      .fit(data)

    val indexedWork = new StringIndexer()
      .setInputCol("work_type")
      .setOutputCol("indexedWork")
      .setHandleInvalid("keep")
      .fit(data)

    val indexedResidence = new StringIndexer()
      .setInputCol("Residence_type")
      .setOutputCol("indexedResidence")
      .setHandleInvalid("keep")
      .fit(data)

    val indexedSmoking = new StringIndexer()
      .setInputCol("smoking_status")
      .setOutputCol("indexedSmoking")
      .setHandleInvalid("keep")
      .fit(data)

    val assembler = new VectorAssembler()
      .setInputCols(Array("id", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"))
      .setOutputCol("features")

    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("Result")

    val trainData = assembler.transform(training)
    val testData = assembler.transform(test)

    // Fit the model
    val lrModel = lr.fit(trainData)

    val accuracy = lrModel.evaluate(testData).accuracy
    print(accuracy)
  }
}




