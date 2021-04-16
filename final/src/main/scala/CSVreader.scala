
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
class CSVreader{

}
import org.apache.spark.sql.SparkSession
object Prediction{
  def main(args: Array[String]): Unit = {
    //create spark object
    val spark = SparkSession.builder()
          .appName("RandomForest")
          .master("local[2]")
          .getOrCreate()
    //get data from csv file
    val healthData = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/healthcare-dataset-stroke-data.csv")
    //healthData.printSchema()
    //healthData.show()
    //Identify the identity column and index column of the entire data set
    //val StringCols = Array()
    val indexedGender = new StringIndexer()
            .setInputCol("gender")
            .setOutputCol("indexedGender")
            .setHandleInvalid("keep")
            .fit(healthData)
    indexedGender.transform(healthData).show();

    val indexedMarried= new StringIndexer()
      .setInputCol("ever_married")
      .setOutputCol("indexedMarried")
      .setHandleInvalid("keep")
      .fit(healthData)
    val indexedWork = new StringIndexer()
      .setInputCol("work_type")
      .setOutputCol("indexedWork")
      .setHandleInvalid("keep")
      .fit(healthData)
    val indexedResidence = new StringIndexer()
      .setInputCol("Residence_type")
      .setOutputCol("indexedResidence")
      .setHandleInvalid("keep")
      .fit(healthData)
    val indexedSmoking = new StringIndexer()
      .setInputCol("smoking_status")
      .setOutputCol("indexedSmoking")
      .setHandleInvalid("keep")
      .fit(healthData)
    val featureCols = Array("indexedGender","age","hypertension","heart_disease","avg_glucose_level")
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
    healthData.show()
    //拆分数据为训练集和测试集（7:3）
    val Array(trainingData, testData) = healthData.randomSplit(Array(0.7, 0.3))
    //testData.show(5)


    //创建模型
    val randomForest = new RandomForestClassifier()
      .setLabelCol("iLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    //使用管道运行转换器和随机森林算法
    val pipeline = new Pipeline()
        .setStages(Array(labelIndexer,indexedGender,featureIndexer,randomForest))
    //训练模型
    val model = pipeline.fit(trainingData)
    //预测
    val predictions = model.transform(testData)
    //输出预测结果
    predictions.select("iLabel", "probability","prediction").show(false)

  }
}
