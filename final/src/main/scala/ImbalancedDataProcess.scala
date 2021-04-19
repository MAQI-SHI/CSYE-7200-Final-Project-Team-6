import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.xx
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.util.Random

object ImbalancedDataProcess {
  def getData={
    /*val spark: SparkSession = SparkSession.builder().appName("test-lightgbm").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val originalData: DataFrame = spark.read.option("header", "true") //第一行作为Schema
      .option("inferSchema", "true") //推测schema类型
      //      .csv("/home/hdfs/hour.csv")
      .csv("src/main/resources/train_strokes.csv")
    originalData*/
  //}
  //def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder().appName("test-lightgbm").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val originalData: DataFrame = spark.read.option("header", "true") //第一行作为Schema
      .option("inferSchema", "true") //推测schema类型
      //      .csv("/home/hdfs/hour.csv")
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

    //map String type to double
    val indexedWork = new StringIndexer()
      .setInputCol("work_type")
      .setOutputCol("indexedWork")
      .setHandleInvalid("keep")
      .fit(originalData)
    val a = indexedWork.transform(originalData)
    /*a.createOrReplaceTempView("p")
    spark.sql("select distinct work_type, indexedwork from p").show()*/
    val new_data = a.na.fill(value="never smoked",Array("smoking_status"))
    val imputer = new Imputer()
      .setInputCols(Array("bmi","avg_glucose_level"))
      .setOutputCols(Array("bmi2","agl2"))
      .setStrategy("mean")

    val data2 = imputer.fit(new_data).transform(new_data)
    //data2.show()
    //new_data.show()
    val indexedSmoking = new StringIndexer()
      .setInputCol("smoking_status")
      .setOutputCol("indexedSmoking")
      .setHandleInvalid("keep")
      .fit(data2)
    val c = indexedSmoking.transform(data2)

    /*c.createOrReplaceTempView("p")
    spark.sql("select distinct smoking_status, indexedSmoking from p").show()*/

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


    /*val vecCols: Array[String] = Array("age","hypertension","heart_disease","indexedMarried","indexedWork",
      "indexedSmoking", "avg_glucose_level", "bmi")*/
    val vecCols: Array[String] = Array("age", "hypertension","indexedWork","agl2",
      "bmi2","indexedSmoking")
    import spark.implicits._
    //原始数据只保留label和features列，追加一列sign标识为老数据
    val inputDF = h.select(labelCol, vecCols: _*).withColumn("sign", lit("O"))
    //inputDF.show(5)
    //需要对最小样本值的数据处理
    val filteredDF = inputDF.filter($"$labelCol" === minSample)
      //filteredDF.show()
    //合并为label和向量列
    val labelAndVecDF = new VectorAssembler()
      .setInputCols(vecCols)
      .setOutputCol("features")
      .setHandleInvalid("keep")
      .transform(filteredDF).select("features")
    //labelAndVecDF.show()
    //转为rdd
    val inputRDD = labelAndVecDF.rdd.map(_.getAs[Vector](0)).repartition(10)

    //smote算法
    val vecRDD: RDD[Vector] = smote(inputRDD, kNei, nNei)


    //以下是公司要求的和之前数据合并
    //生成dataframe，将向量列展开，追加一列sign标识为新数据
    val vecDF: DataFrame = vecRDD.map(vec => (1, vec.toArray)).toDF(labelCol, "features")
    println(1)
    val newCols = (0 until vecCols.size).map(i => $"features".getItem(i).alias(vecCols(i)))
    println(2)
    //根据需求，新数据应该为样本量*n，当前测试数据label为0的样本量为5514，则会新增5514*10=55140
    val newDF = vecDF.select(($"$labelCol" +: newCols): _*).withColumn("sign", lit("N"))
    //newDF.show(5)
    import org.apache.spark.sql.types._

    newDF.createOrReplaceTempView("newDF")
    val newDF2 = spark.sql("SELECT stroke, CAST(age AS DECIMAL(10,0)) , CAST(hypertension AS DECIMAL(10,0)), " +
      "CAST(indexedWork AS DECIMAL(10,0)), CAST(agl2 AS DECIMAL(10,2)), CAST(bmi2 AS DECIMAL(10,2)), " +
      "CAST(indexedSmoking AS DECIMAL(10,0)), sign FROM newDF")
    //newDF2.show()
    //和原数据合并
    val finalDF = inputDF.union(newDF2)
    finalDF.show

    import scala.collection.JavaConversions._
    //查看原数据
    val aggSeq: Seq[Row] = h.groupBy(labelCol).agg(count(labelCol).as("labelCount"))
      .collectAsList().toSeq
    //println(aggSeq)

    //查看平衡后数据，根据需求，则最终合并后，label为0的样本为55140+5514=60654
    val aggSeq1: Seq[Row] = finalDF.groupBy(labelCol).agg(count(labelCol).as("labelCount"))
      .collectAsList().toSeq
    //println(aggSeq1)
    finalDF
  }

  //smote
  def smote(data: RDD[Vector], k: Int, N: Int): RDD[Vector] = {
    val vecAndNeis: RDD[(Vector, Array[Vector])] = data.mapPartitions(iter => {
      val vecArr: Array[Vector] = iter.toArray
      //对每个分区的每个vector产生笛卡尔积
      val cartesianArr: Array[(Vector, Vector)] = vecArr.flatMap(vec1 => {
        vecArr.map(vec2 => (vec1, vec2))
      }).filter(tuple => tuple._1 != tuple._2)
      cartesianArr.groupBy(_._1).map { case (vec, vecArr) => {
        (vec, vecArr.sortBy(x => Vectors.sqdist(x._1, x._2)).take(k).map(_._2))
      }
      }.iterator
    })
    //1.从这k个近邻中随机挑选一个样本，以该随机样本为基准生成N个新样本
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
