class ImbData {

}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.xx
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.util.Random

/**
 * Created By TheBigBlue on 2020/3/23
 * Description :
 */
object ImbalancedDataProcess {

  //def main(args: Array[String]): Unit = {
  def getData={
    val spark: SparkSession = SparkSession.builder().appName("test-lightgbm").master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val originalData: DataFrame = spark.read.option("header", "true") //第一行作为Schema
      .option("inferSchema", "true") //推测schema类型
      //      .csv("/home/hdfs/hour.csv")
      .csv("src/main/resources/train_strokes.csv")

    val kNei = 4
    val nNei = 10
    //少数样本值
    val minSample = 1
    //标签列
    val labelCol = "stroke"

    val indexedWork = new StringIndexer()
      .setInputCol("work_type")
      .setOutputCol("indexedWork")
      .setHandleInvalid("keep")
      .fit(originalData)
    val a = indexedWork.transform(originalData)
    val indexedResidence = new StringIndexer()
      .setInputCol("Residence_type")
      .setOutputCol("indexedResidence")
      .setHandleInvalid("keep")
      .fit(a)
    val b = indexedResidence.transform(a)
    val new_data = b.na.fill(value="never smoked",Array("smoking_status"))
    //new_data.show()
    val indexedSmoking = new StringIndexer()
      .setInputCol("smoking_status")
      .setOutputCol("indexedSmoking")
      .setHandleInvalid("keep")
      .fit(new_data)
    val c = indexedSmoking.transform(new_data)
    val indexedGender = new StringIndexer()
      .setInputCol("gender")
      .setOutputCol("indexedGender")
      .setHandleInvalid("keep")
      .fit(c)
    val indexedMarried= new StringIndexer()
      .setInputCol("ever_married")
      .setOutputCol("indexedMarried")
      .setHandleInvalid("keep")
      .fit(c)
    val d = indexedGender.transform(c)
    val e = indexedMarried.transform(d)
    //e.show(5)
    val h = e.drop("id","gender","ever_married","work_type","Residence_type","smoking_status")
    //h.show()
    // 连续列
    val vecCols: Array[String] = Array("indexedWork", "indexedResidence", "avg_glucose_level", "bmi")

    import spark.implicits._
    //原始数据只保留label和features列，追加一列sign标识为老数据
    val inputDF = h.select(labelCol, vecCols: _*).withColumn("sign", lit("O"))
    //inputDF.show()
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
    println(3)

    //smote算法
    val vecRDD: RDD[Vector] = smote(inputRDD, kNei, nNei)


    //以下是公司要求的和之前数据合并
    //生成dataframe，将向量列展开，追加一列sign标识为新数据
    val vecDF: DataFrame = vecRDD.map(vec => (1, vec.toArray)).toDF(labelCol, "features")
    val newCols = (0 until vecCols.size).map(i => $"features".getItem(i).alias(vecCols(i)))
    //根据需求，新数据应该为样本量*n，当前测试数据label为0的样本量为5514，则会新增5514*10=55140
    val newDF = vecDF.select(($"$labelCol" +: newCols): _*).withColumn("sign", lit("N"))
    //newDF.show()
    //和原数据合并
    val finalDF = inputDF.union(newDF)
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
