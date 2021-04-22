
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

class Model extends AnyFlatSpec with Matchers {
  val RFacc = RF2.Run()
  val DTacc = DT.Run()
  val LRacc = LR.Run()

  "RF Model accuracy" should "greater than 0.7" in {
    RFacc.apply(0) should be > 0.7
  }

  "RF Model isStroke accuracy" should "greater" in {
    RFacc.apply(1) should be > 0.0
  }

  "RF Model isNotStroke accuracy" should "greater than 0.7" in {
    RFacc.apply(2) should be > 0.7
  }

  "LR Model accuracy" should "greater than 0.7" in {

    LRacc.apply(0) should be > 0.7
  }

  "DT Model accuracy" should "greater than 0.7" in {

    DTacc.apply(0) should be > 0.7
  }

  "DT Model isStroke accuracy" should "greater" in {

    DTacc.apply(1) should be > 0.0
  }

  "DT Model isNotStroke accuracy" should "greater than 0.7" in {

    DTacc.apply(2) should be > 0.7
  }
}
