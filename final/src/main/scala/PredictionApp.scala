import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import scala.util.control.Breaks.{break, breakable}
import java.util.Date
object PredictionApp extends App {
  val spark = SparkSession.builder()
    .appName("RandomForest")
    .master("local[2]")
    .getOrCreate()
  val model = PipelineModel.load("./rfModel")
  //val Array(trainingData, testData) = ImbalancedDataProcess.getData.randomSplit(Array(0.9, 0.1))
  breakable {
    while (true) {
      println("Press 1 to predict, else will close the app.")
      val condition = scala.io.StdIn.readInt()
      if (condition == 1) {
        println("Please input your age:")
        val age = scala.io.StdIn.readInt()

        println("Please input 1 if your have hypertension or 0 don't have hypertension:")
        val hyp = scala.io.StdIn.readInt()

        println("Please input 1 if your have hypertension or 0 don't have hypertension:")
        val work = scala.io.StdIn.readInt()

        println("Please input 1 if your have hypertension or 0 don't have hypertension:")
        val agl2 = scala.io.StdIn.readInt()

        println("Please input 1 if your have hypertension or 0 don't have hypertension:")
        val bmi = scala.io.StdIn.readInt()

        println("Please input 1 if your have hypertension or 0 don't have hypertension:")
        val smo = scala.io.StdIn.readInt()
        /**
         * Timestamp to record the start time of predicting system*/
        val prev = new Date()

        //Convert user inputs into required dataframe
        val df = spark.createDataFrame(Seq(
          (age, hyp, work, agl2, bmi, smo),
        )).toDF("age","hypertension","indexedWork",
          "agl2","bmi2","indexedSmoking")

        val assembler = new VectorAssembler()
          .setInputCols(Array("age","hypertension","indexedWork",
            "agl2","bmi2","indexedSmoking"))
          .setOutputCol("features")

        val validData = assembler.transform(df)

        //Predict user inputs with selected best model
        val predictions = model.transform(validData)
        predictions.show(false)

        //Timestamp to record the end time of predicting system
        val now = new Date()

        //Show Predicted results including response time and final result
        println("Pridict Time: " + ((now.getTime - prev.getTime).toDouble / 1000))

        val finalRes = predictions.select("prediction").rdd.first().getDouble(0)

        if (finalRes == 1.0) {
          println("You have stroke!")
        } else {
          println("You don't have stroke!")
        }
      } else {
        break
      }
    }
  }

}
