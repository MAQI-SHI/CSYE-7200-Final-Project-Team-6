
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

class ScalaLR{

}

object logic {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("LR")
      .master("local[2]")
      .getOrCreate()

    val healthData = ImbalancedDataProcess.getData

    val featureCols = Array("indexedResidence","indexedWork",
      "avg_glucose_level","bmi")

    val featureIndexer = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("indexedFeatures")
      .setHandleInvalid("keep")
    //featureIndexer.transform(healthData).show()

    val labelIndexer = new StringIndexer()
      .setInputCol("stroke")
      .setOutputCol("iLabel")
      .setHandleInvalid("keep")
      .fit(healthData)

    healthData.createOrReplaceTempView("pos")

    val postive = spark.sql("select * from pos where stroke = 1")
    val nagetive = spark.sql("select * from pos where stroke = 0")

    val Array(trainingDatap, testDatap) = postive.randomSplit(Array(0.7, 0.3))
    val Array(trainingDatan, testDatan) = nagetive.randomSplit(Array(0.7, 0.3))
    testDatan.show(5)
    testDatap.show(5)
    val trainingData = trainingDatap.union(trainingDatan)
    val testData = testDatap.union(testDatan)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("iLabel")
      .setFeaturesCol("indexedFeatures")

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, lr))

    val model = pipeline.fit(trainingData)
    model.write.overwrite().save("./lrModel")

    // Fit the model
    val predictions = model.transform(testData)

    predictions.createOrReplaceTempView("p")

    val isStroke = spark.sql("select * from p where iLabel = 1")
    val notStroke = spark.sql("select * from p where iLabel = 0")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("iLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    val isStrokeAccuracy = evaluator.evaluate(isStroke)
    val notStrokeAccuracy = evaluator.evaluate(notStroke)
    println(s"stroke accuracy = ${isStrokeAccuracy}")
    println(s"not stroke accuracy = ${notStrokeAccuracy}")
    println(s"accuracy = ${accuracy}")
  }
}




