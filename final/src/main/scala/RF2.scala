import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession


object RF2{
  def main(args: Array[String]): Unit = {
    //create spark object
    val spark = SparkSession.builder()
      .appName("RandomForest")
      .master("local[2]")
      .getOrCreate()
    //get data from csv file
    val healthData = ImbalancedDataProcess.getData
    //healthData.printSchema()
    //healthData.show()
    //Identify the identity column and index column of the entire data set

    val featureCols = Array("age","hypertension","indexedWork",
      "agl2","bmi2","indexedSmoking")
    /*val featureCols = Array("age",
      "hypertension","heart_disease","avg_glucose_level")*/

    //设置树的最大层次
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

    //拆分数据为训练集和测试集（7:3）
    healthData.createOrReplaceTempView("pos")

    val postive = spark.sql("select * from pos where stroke = 1")
    val nagetive = spark.sql("select * from pos where stroke = 0")

    val Array(trainingDatap, testDatap) = postive.randomSplit(Array(0.7, 0.3))
    val Array(trainingDatan, testDatan) = nagetive.randomSplit(Array(0.7, 0.3))
    testDatan.show(5)
    testDatap.show(5)
    val trainingData = trainingDatap.union(trainingDatan)
    val testData = testDatap.union(testDatan)

    //testData.show(5)


    //创建模型
    val randomForest = new RandomForestClassifier()
      .setLabelCol("iLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setSeed(4)

    //使用管道运行转换器和随机森林算法
    /*val pipeline = new Pipeline()
      .setStages(Array(labelIndexer,
        featureIndexer,
        randomForest))*/

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, randomForest))

    //训练模型
    val model = pipeline.fit(trainingData)
    model.write.overwrite().save("./rfModel")
    //trainingData.show(5)
    //预测
    val predictions = model.transform(testData)
    //testData.show(5)
    //输出预测结果
    predictions.select("iLabel", "probability","prediction").show(30,false)

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
