import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import scala.util.control.Breaks.{break, breakable}
import java.util.Date
object PredictionApp extends App {
  /**
   * create spark object
   */
  val spark = SparkSession.builder()
    .appName("PredictionApp")
    .master("local[2]")
    .getOrCreate()
  /**
   * choose the model, according to the accuracy, the decision tree is the best model
   */
  val model = PipelineModel.load("./dtModel")
  breakable {
    while (true) {
      println("Press 1 to predict, else will close the app.")
      val condition = scala.io.StdIn.readInt()
      if (condition == 1) {
        println("Please input your age:")
        val age = scala.io.StdIn.readDouble()

        println("Please input 1 if your have hypertension or 0 don't have hypertension:")
        val hyp = scala.io.StdIn.readDouble()

        println("Please input your work_type,0 for Private, 1 for Self-employed, 2 for children, 3 for Govt_job, 4 " +
          "for Never_worked:")
        val work = scala.io.StdIn.readDouble()

        println("Please input your avg_glucose_level:")
        val agl2 = scala.io.StdIn.readDouble()

        println("Please input your bmi:")
        val bmi = scala.io.StdIn.readDouble()

        println("Please input your smoking_status, 0 for never smoked, 1 for formerly smoked, 2 for smokes:")
        val smo = scala.io.StdIn.readDouble()
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
          println("You will have stroke!")
        } else {
          println("You will not have stroke!")
        }
      } else {
        break
      }
    }
  }

}
