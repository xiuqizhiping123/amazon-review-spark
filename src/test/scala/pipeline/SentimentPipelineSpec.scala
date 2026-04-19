package pipeline

import domin.Review
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SentimentPipelineSpec extends AnyFlatSpec with Matchers {

  private val dict = Afinn.load()

  "scoreReview" should "assign label 1.0 for rating >= 4.0" in {
    val review = Review(5.0, "great product", "title")
    val scored = SentimentPipeline.scoreReview(dict)(review)
    scored.label shouldBe 1.0
  }

  it should "assign label 0.0 for rating < 4.0" in {
    val review = Review(2.0, "terrible product", "title")
    val scored = SentimentPipeline.scoreReview(dict)(review)
    scored.label shouldBe 0.0
  }

  it should "correctly count positive and negative words" in {
    val review = Review(5.0, "good great bad", "title")
    val scored = SentimentPipeline.scoreReview(dict)(review)
    scored.positiveCount shouldBe 2
    scored.negativeCount shouldBe 1
  }

  it should "return zero sentiment score for unknown words" in {
    val review = Review(5.0, "xyzabc qwerty", "title")
    val scored = SentimentPipeline.scoreReview(dict)(review)
    scored.sentimentScore shouldBe 0
  }

  it should "ignore punctuation when tokenizing" in {
    val review = Review(5.0, "good, great!", "title")
    val scored = SentimentPipeline.scoreReview(dict)(review)
    scored.positiveCount shouldBe 2
  }

  it should "be case insensitive" in {
    val lower = SentimentPipeline.scoreReview(dict)(Review(5.0, "good", ""))
    val upper = SentimentPipeline.scoreReview(dict)(Review(5.0, "GOOD", ""))
    lower.sentimentScore shouldBe upper.sentimentScore
  }

  "featureCols" should "return only lexicon columns for LexiconOnly" in {
    SentimentPipeline.featureCols(SentimentPipeline.LexiconOnly) shouldBe
      Array("sentimentScore", "wordCount", "positiveCount", "negativeCount")
  }

  it should "return only tfidf column for TfidfOnly" in {
    SentimentPipeline.featureCols(SentimentPipeline.TfidfOnly) shouldBe
      Array("tfidfFeatures")
  }

  it should "return all columns for Hybrid" in {
    val cols = SentimentPipeline.featureCols(SentimentPipeline.Hybrid)
    cols should contain("tfidfFeatures")
    cols should contain("sentimentScore")
  }
}