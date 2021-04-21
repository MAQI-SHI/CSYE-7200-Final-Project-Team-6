import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.xx
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.util.Random

object ImbalancedDataProcess extends App {
def getData={
    //create spark
    val spark: SparkSession = SparkSession.builder().appName("test-lightgbm").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    //read csv file
    val originalData: DataFrame = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/train_strokes.csv")
    //k refers to the k nearest neighbor with the closest Euclidean distance,
    // But only one of the nearest neighbors is randomly selected, and based on this,
    // n new samples are selected on this line.
    val kNei = 5
    val nNei = 10
    //Minority sample value
    val minSample = 1
    //label column
    val labelCol = "stroke"

    //map String type to double type
    val indexedWork = new StringIndexer()
      .setInputCol("work_type")
      .setOutputCol("indexedWork")
      .setHandleInvalid("keep")
      .fit(originalData)
    val a = indexedWork.transform(originalData)
    //replace na data with mode
    val new_data = a.na.fill(value="never smoked",Array("smoking_status"))
    //replace na with average data
    val imputer = new Imputer()
      .setInputCols(Array("bmi","avg_glucose_level"))
      .setOutputCols(Array("bmi2","agl2"))
      .setStrategy("mean")
    val data2 = imputer.fit(new_data).transform(new_data)
    val indexedSmoking = new StringIndexer()
      .setInputCol("smoking_status")
      .setOutputCol("indexedSmoking")
      .setHandleInvalid("keep")
      .fit(data2)
    val c = indexedSmoking.transform(data2)
    val indexedMarried= new StringIndexer()
      .setInputCol("ever_married")
      .setOutputCol("indexedMarried")
      .setHandleInvalid("keep")
      .fit(c)
    val e = indexedMarried.transform(c)
    //e.show(5)

    //drop useless column
    val h = e.drop("gender","ever_married","work_type","Residence_type","smoking_status")
    //h.show()

    //set feature column
    val vecCols: Array[String] = Array("age", "hypertension","indexedWork","agl2", "bmi2","indexedSmoking")
    import spark.implicits._
    //The original data only retains the label and features columns, and a column of sign is added as the old data
    val inputDF = h.select(labelCol, vecCols: _*).withColumn("sign", lit("O"))
    //inputDF.show(5)

    //Need data processing for the smallest sample value
    val filteredDF = inputDF.filter($"$labelCol" === minSample)
      //filteredDF.show()

    //Combine into label and vector column
    val labelAndVecDF = new VectorAssembler()
      .setInputCols(vecCols)
      .setOutputCol("features")
      .setHandleInvalid("keep")
      .transform(filteredDF).select("features")
    //labelAndVecDF.show()

    //change into rdd
    val inputRDD = labelAndVecDF.rdd.map(_.getAs[Vector](0)).repartition(10)

    //smote algorithm
    val vecRDD: RDD[Vector] = smote(inputRDD, kNei, nNei)

    //Generate a dataframe, expand the vector column, and add a column of sign as new data
    val vecDF: DataFrame = vecRDD.map(vec => (1, vec.toArray)).toDF(labelCol, "features")

    val newCols = (0 until vecCols.size).map(i => $"features".getItem(i).alias(vecCols(i)))


    val newDF = vecDF.select(($"$labelCol" +: newCols): _*).withColumn("sign", lit("N"))
    //newDF.show(5)

    //set the format of the new data with the old data
    newDF.createOrReplaceTempView("newDF")
    val newDF2 = spark.sql("SELECT stroke, CAST(age AS DECIMAL(10,0)) , CAST(hypertension AS DECIMAL(10,0)), " +
      "CAST(indexedWork AS DECIMAL(10,0)), CAST(agl2 AS DECIMAL(10,2)), CAST(bmi2 AS DECIMAL(10,2)), " +
      "CAST(indexedSmoking AS DECIMAL(10,0)), sign FROM newDF")
    //newDF2.show()
    //union new data with old data
    val finalDF = inputDF.union(newDF2)
    finalDF.show

    import scala.collection.JavaConversions._
    //check final data
    val aggSeq: Seq[Row] = h.groupBy(labelCol).agg(count(labelCol).as("labelCount"))
      .collectAsList().toSeq
    //println(aggSeq)
    val aggSeq1: Seq[Row] = finalDF.groupBy(labelCol).agg(count(labelCol).as("labelCount"))
      .collectAsList().toSeq
    //println(aggSeq1)
    //return final data
    finalDF
}

  //smote
  def smote(data: RDD[Vector], k: Int, N: Int): RDD[Vector] = {
    val vecAndNeis: RDD[(Vector, Array[Vector])] = data.mapPartitions(iter => {
      val vecArr: Array[Vector] = iter.toArray
      //Generate Cartesian product for each vector in each partition
      val cartesianArr: Array[(Vector, Vector)] = vecArr.flatMap(vec1 => {
        vecArr.map(vec2 => (vec1, vec2))
      }).filter(tuple => tuple._1 != tuple._2)
      cartesianArr.groupBy(_._1).map { case (vec, vecArr) => {
        (vec, vecArr.sortBy(x => Vectors.sqdist(x._1, x._2)).take(k).map(_._2))
      }
      }.iterator
    })
    //Randomly select a sample from the k nearest neighbors, and generate N new samples based on the random sample
    val vecRDD = vecAndNeis.flatMap { case (vec, neighbours) =>
      (1 to N).map { i =>
        val newK = if (k > neighbours.size) neighbours.size else k
        val rn = neighbours(Random.nextInt(newK))
        val diff = rn.copy
        xx.BLAS.axpy(-1.0, vec, diff)
        val newVec = vec.copy
        xx.BLAS.axpy(Random.nextDouble(), diff, newVec)
        newVec
      }.iterator
    }
    vecRDD
  }
}
