import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import java.util.Date
object PredictionApp extends App {
  val spark = SparkSession.builder()
    .appName("RandomForest")
    .master("local[2]")
    .getOrCreate()

  val Array(trainingData, testData) = ImbalancedDataProcess.getData.randomSplit(Array(0.9, 0.1))
  //val testData = ImbalancedDataProcess.getData.randomSplit(0.01)
  val start_time =new Date().getTime
  val model = PipelineModel.load("./rfModel")
  val predictions = model.transform(testData)
  val end_time =new Date().getTime
  //testData.show(5)
  //输出预测结果
  //predictions.select("iLabel", "probability","prediction").show(30,false)

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
  println((end_time-start_time)/1000)
}
