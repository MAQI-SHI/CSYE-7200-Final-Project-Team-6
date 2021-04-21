
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

class Model extends AnyFlatSpec with Matchers {
  "RF Model accuracy" should "greater than 0.7" in {
    val RFacc = RF2.Run()
    RFacc should be > 0.7
  }

  "LR Model accuracy" should "greater than 0.7" in {
    val LRacc = LR.Run()
    LRacc should be > 0.7
  }

  "DT Model accuracy" should "greater than 0.6" in {
    val DTacc = DT.Run()
    DTacc should be > 0.6
  }
}
