import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * this object is for analyze the data correlation
 */
object Correlaction extends App {
  /**
   * create spark object
   */
  val spark: SparkSession = SparkSession.builder().appName("test-lightgbm").master("local[4]").getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  /**
   * read csv file
   */
  val originalData: DataFrame = spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv("src/main/resources/train_strokes.csv")
  /**
   * map String type to Double type
   */
  val indexedGender = new StringIndexer()
    .setInputCol("gender")
    .setOutputCol("indexedGender")
    .setHandleInvalid("keep")
    .fit(originalData)
  val data1 = indexedGender.transform(originalData)
  val indexedMarried= new StringIndexer()
    .setInputCol("ever_married")
    .setOutputCol("indexedMarried")
    .setHandleInvalid("keep")
    .fit(data1)
  val data2 = indexedMarried.transform(data1)
  val indexedWork = new StringIndexer()
    .setInputCol("work_type")
    .setOutputCol("indexedWork")
    .setHandleInvalid("keep")
    .fit(data2)
  val data3 = indexedWork.transform(data2)
  val indexedResidence = new StringIndexer()
    .setInputCol("Residence_type")
    .setOutputCol("indexedResidence")
    .setHandleInvalid("keep")
    .fit(data3)
  val data4 = indexedResidence.transform(data3)
  val indexedSmoking = new StringIndexer()
    .setInputCol("smoking_status")
    .setOutputCol("indexedSmoking")
    .setHandleInvalid("keep")
    .fit(data4)
  val data5 = indexedSmoking.transform(data4)
  /**
   * Read and convert data.
   */
  val rddpreD = data5.rdd.map{row =>
    val first = row.getAs[Double]("indexedGender")
    val second = row.getAs[Double]("age")
    val third = row.getAs[Integer]("hypertension")
    val fourth = row.getAs[Integer]("heart_disease")
    val fifth= row.getAs[Double]("indexedMarried")
    val sixth= row.getAs[Double]("indexedWork")
    val seventh= row.getAs[Double]("indexedResidence")
    val eighth= row.getAs[Double]("avg_glucose_level")
    val ninth= row.getAs[Double]("bmi")
    val tenth= row.getAs[Double]("indexedSmoking")
    val eleth= row.getAs[Integer]("stroke")
    Vectors.dense(first.toDouble,second.toDouble,third.toDouble,fourth.toDouble,fifth.toDouble,sixth.toDouble,seventh
      .toDouble,eighth.toDouble,ninth.toDouble,tenth.toDouble,eleth.toDouble)
  }

  /**
   * Compute correlation.
   */
  val correlMatrix = Statistics.corr(rddpreD)
  //println(correlMatrix.toString(10,Int.MaxValue))
  /**
   * Convert correlation matrix to dataframe
   */
  import spark.implicits._
  val cor = (0 until correlMatrix.numCols)
  val df = correlMatrix.transpose
    .colIter.toSeq
    .map(_.toArray)
    .toDF("arr")

  val correlation = cor.foldLeft(df)((df, i) => df.withColumn("_" + (i+1), $"arr"(i)))
    .drop("arr")
    .withColumnRenamed("_1","indexedGender")
    .withColumnRenamed("_2","age")
    .withColumnRenamed("_3","hypertension")
    .withColumnRenamed("_4","heart_disease")
    .withColumnRenamed("_5","indexedMarried")
    .withColumnRenamed("_6","indexedWork")
    .withColumnRenamed("_7","indexedResidence")
    .withColumnRenamed("_8","avg_glucose_level")
    .withColumnRenamed("_9","bmi")
    .withColumnRenamed("_10","indexedSmoking")
    .withColumnRenamed("_11","stroke")
  correlation.show(false)
  correlation.createOrReplaceTempView("cor")
  val correlation2 = spark.sql("SELECT CAST(indexedGender AS DECIMAL(10,2)), CAST(age AS DECIMAL(10,2))," +
    "CAST(hypertension AS DECIMAL(10,2)), CAST(heart_disease AS DECIMAL(10,2)), CAST(indexedMarried AS DECIMAL(10,2))" +
    ", CAST(indexedWork AS DECIMAL(10,2)), CAST(indexedResidence AS DECIMAL(10,2)), CAST(avg_glucose_level AS DECIMAL" +
    "(10,2)),CAST(bmi AS DECIMAL(10,2)), CAST(indexedSmoking AS DECIMAL(10,2)),CAST(stroke AS DECIMAL(10,2)) FROM cor")
  correlation2.show()
}
