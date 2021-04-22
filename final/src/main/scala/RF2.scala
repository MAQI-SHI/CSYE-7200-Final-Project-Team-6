import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession


object RF2{
  def Run() = {
    /**
     * create spark object
     */
    val spark = SparkSession.builder()
      .appName("RandomForest")
      .master("local[2]")
      .getOrCreate()

    /**
     * get data
     */
    val healthData = ImbalancedDataProcess.getData

    /**
     * set feature column and label colum
     */
    val featureCols = Array("age","hypertension","indexedWork",
      "agl2","bmi2","indexedSmoking")
    val featureIndexer = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("indexedFeatures")
      .setHandleInvalid("keep")

    val labelIndexer = new StringIndexer()
      .setInputCol("stroke")
      .setOutputCol("iLabel")
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
     * create RandomForest model
     */
    val randomForest = new RandomForestClassifier()
      .setLabelCol("iLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(4)

    /**
     * use pipeline and randomForest algorithm
     */
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, randomForest))

    /**
     * train and save model
     */
    val model = pipeline.fit(trainingData)
    model.write.overwrite().save("./rfModel")

    /**
     * test model
     */
    val predictions = model.transform(testData)

    /**
     * result
     * Calculate the prediction accuracy of stroke and non-stroke separately
     */
    predictions.select("iLabel", "probability","prediction").show(30,false)
    predictions.createOrReplaceTempView("p")
    val isStroke = spark.sql("select * from p where iLabel = 1")
    val right = spark.sql("select * from p where iLabel = 1 and prediction = 1 and sign = 'O'")
    val notStroke = spark.sql("select * from p where iLabel = 0")

    /**
     * create evaluator
     */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("iLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    /**
     * print result
     */
    val accuracy = evaluator.evaluate(predictions)
    val isStrokeAccuracy = evaluator.evaluate(isStroke)
    val notStrokeAccuracy = evaluator.evaluate(notStroke)
    println("RandomForest result")
    println(s"stroke accuracy = ${isStrokeAccuracy}")
    println(s"not stroke accuracy = ${notStrokeAccuracy}")
    println(s"accuracy = ${accuracy}")
    val acArray = Array(accuracy,isStrokeAccuracy,notStrokeAccuracy)
    acArray
  }

}
