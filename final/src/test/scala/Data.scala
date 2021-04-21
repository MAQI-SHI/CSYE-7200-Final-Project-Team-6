
import ImbalancedDataProcess._
import RF2._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.SparkSession

class Data extends AnyFlatSpec with Matchers {

  val spark = SparkSession.builder()
    .appName("RandomForest")
    .master("local[2]")
    .getOrCreate()

  behavior of "dataset"
  it should "data count accuracy" in {
    val finalData = getData.count()
    getData.createOrReplaceTempView("p")
    val generatedData = spark.sql("select * from p where stroke = 1 and sign = 'N'").count()
    finalData shouldBe 51230
    generatedData shouldBe 7830
  }
}
