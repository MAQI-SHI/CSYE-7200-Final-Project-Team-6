
import ImbalancedDataProcess._
import RF2._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{sum, col}
class Data extends AnyFlatSpec with Matchers {

  val spark = SparkSession.builder()
    .appName("RandomForest")
    .master("local[2]")
    .getOrCreate()

  val finalData = getData
  val finalDataNumber = finalData.count()
  getData.createOrReplaceTempView("p")
  val generatedData = spark.sql("select * from p where stroke = 1 and sign = 'N'").count()
  it should "data count accuracy" in {

    finalDataNumber shouldBe 51230
    generatedData shouldBe 7830
  }

  it should "null data count" in{
    val naData = finalData.select(finalData.columns.map(c => sum(col(c).isNull.cast("Double")).alias(c)): _*)
    naData.show()
    val naDataNum = naData.count()
    naDataNum shouldBe 1
  }
}
