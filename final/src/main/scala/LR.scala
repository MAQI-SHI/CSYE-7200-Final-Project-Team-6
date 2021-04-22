import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

/**
 * this object is for Logistic Regression model
 */
object LR extends App{
  def Run()= {
    /**
     * create spark object
     */
    val spark = SparkSession.builder()
      .appName("LR")
      .master("local[2]")
      .getOrCreate()
    /**
     * get data
     */
    val healthData = ImbalancedDataProcess.getData

    /**
     * set feature column and label colum
     */
    val featureCols = Array("age","hypertension","indexedWork", "agl2","bmi2","indexedSmoking")

    val featureIndexer = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("indexedFeatures")
      .setHandleInvalid("keep")

    val labelIndexer = new StringIndexer()
      .setInputCol("stroke")
      .setOutputCol("indexedLabel")
      .setHandleInvalid("keep")
      .fit(healthData)

    /**
     * Split data into training set and test set (7:3)
     * select data from stroke and not stroke separately
     */
    healthData.createOrReplaceTempView("pos")
    val postive = spark.sql("select * from pos where stroke = 1")
    val nagetive = spark.sql("select * from pos where stroke = 0")

    val Array(trainingDatap, testDatap) = postive.randomSplit(Array(0.7, 0.3))
    val Array(trainingDatan, testDatan) = nagetive.randomSplit(Array(0.7, 0.3))

    val trainingData = trainingDatap.union(trainingDatan)
    val testData = testDatap.union(testDatan)

    /**
     * create Logistic Regression model
     */
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, lr))

    val model = pipeline.fit(trainingData)
    model.write.overwrite().save("./lrModel")

    /**
     * fit the model
     */
    val predictions = model.transform(testData)

    /**
     * result
     * Calculate the prediction accuracy of stroke and non-stroke separately
     */
    predictions.createOrReplaceTempView("p")

    predictions.select("indexedLabel", "probability", "prediction").show(30,false)

    val isStroke = spark.sql("select * from p where indexedLabel = 1")
    val notStroke = spark.sql("select * from p where indexedLabel = 0")

    /**
     * create evaluator
     */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    val isStrokeAccuracy = evaluator.evaluate(isStroke)
    val notStrokeAccuracy = evaluator.evaluate(notStroke)
    println("Logistic Regression result")
    println(s"stroke accuracy = ${isStrokeAccuracy}")
    println(s"not stroke accuracy = ${notStrokeAccuracy}")
    println(s"accuracy = ${accuracy}")
    val acArray = Array(accuracy,isStrokeAccuracy,notStrokeAccuracy)
    acArray
  }
}
