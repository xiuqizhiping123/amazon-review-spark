package pipeline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AfinnSpec extends AnyFlatSpec with Matchers {

  private val dict = Afinn.load()

  "Afinn.load" should "return a non-empty dictionary" in {
    dict should not be empty
  }

  it should "contain known positive words" in {
    dict.get("good") shouldBe Some(3)
    dict.get("love") shouldBe Some(3)
  }

  it should "contain known negative words" in {
    dict.get("bad") shouldBe Some(-3)
    dict.get("horrible") shouldBe Some(-3)
  }

  it should "not contain stopwords" in {
    dict.get("the") shouldBe None
    dict.get("and") shouldBe None
  }
}