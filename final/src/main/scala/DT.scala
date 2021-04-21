import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession

object DT extends App{
  def Run()= {
    val spark = SparkSession.builder()
      .appName("RandomForest")
      .master("local[2]")
      .getOrCreate()
    //get data from csv file
    val healthData = ImbalancedDataProcess.getData

    val labelIndexer = new StringIndexer()
      .setInputCol("stroke")
      .setOutputCol("indexedLabel")
      .setHandleInvalid("keep")
      .fit(healthData)

    val featureCols = Array("age","hypertension","indexedWork", "agl2","bmi2","indexedSmoking")

    val featureIndexer = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("indexedFeatures")
      .setHandleInvalid("keep")

    //拆分数据为训练集和测试集（7:3）
    healthData.createOrReplaceTempView("pos")

    val postive = spark.sql("select * from pos where stroke = 1")
    val nagetive = spark.sql("select * from pos where stroke = 0")
    val real = spark.sql("select * from pos where stroke = 1 and sign = '0'")

    val Array(trainingDatap, testDatap) = postive.randomSplit(Array(0.7, 0.3))
    val Array(trainingDatan, testDatan) = nagetive.randomSplit(Array(0.7, 0.3))
    testDatan.show(5)
    testDatap.show(5)

    val trainingData = trainingDatap.union(trainingDatan)
    val testData = testDatap.union(testDatan)

    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val model = pipeline.fit(trainingData)
    model.write.overwrite().save("./dtModel")

    val predictions = model.transform(testData)

    predictions.select("indexedLabel", "probability", "prediction").show(30,false)

    predictions.createOrReplaceTempView("p")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val isStroke = spark.sql("select * from p where indexedLabel = 1")
    val right = spark.sql("select * from p where indexedLabel = 1 and prediction = 1 and sign = 'O'")
    right.show()
    val notStroke = spark.sql("select * from p where indexedLabel = 0")

    val accuracy = evaluator.evaluate(predictions)
    val isStrokeAccuracy = evaluator.evaluate(isStroke)
    val notStrokeAccuracy = evaluator.evaluate(notStroke)
    println(s"stroke accuracy = ${isStrokeAccuracy}")
    println(s"not stroke accuracy = ${notStrokeAccuracy}")
    println(s"accuracy = ${accuracy}")

    accuracy
  }
}
