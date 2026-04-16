package pipeline

import domin.{Review, ScoredReview}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object SentimentPipeline {
  val tfidfCol = "tfidfFeatures"
  val lexiconCols = Array("sentimentScore", "wordCount", "positiveCount", "negativeCount")
  val hybridCols: Array[String] = lexiconCols :+ tfidfCol

  sealed trait FeatureMode

  case object LexiconOnly extends FeatureMode

  case object TfidfOnly extends FeatureMode

  case object Hybrid extends FeatureMode

  def featureCols(mode: FeatureMode): Array[String] = mode match {
    case LexiconOnly => lexiconCols
    case TfidfOnly => Array(tfidfCol)
    case Hybrid => hybridCols
  }

  private def tokenizeRaw(text: String): Array[String] = {
    text.toLowerCase
      .replaceAll("[^a-zA-Z\\s]", "")
      .split("\\s+")
      .filter(_.nonEmpty)
  }

  def scoreReview(dict: Map[String, Int])(review: Review): ScoredReview = {
    val tokens = tokenizeRaw(review.text)
    val scores = tokens.flatMap(dict.get)
    val pos = scores.count(_ > 0)
    val neg = scores.count(_ < 0)
    val total = scores.sum
    val label = if (review.rating >= 4.0) 1.0 else 0.0
    ScoredReview(review.rating, review.text, total, tokens.length, pos, neg, label)
  }

  def prepareText(df: DataFrame): DataFrame =
    df.withColumn("cleanText", lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", "")))

  def buildTfidfStages(): Array[PipelineStage] = Array(
    new Tokenizer()
      .setInputCol("cleanText")
      .setOutputCol("tokens"),
    new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filteredTokens"),
    new HashingTF()
      .setInputCol("filteredTokens")
      .setOutputCol("rawFeatures")
      .setNumFeatures(10000),
    new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol(tfidfCol)
  )

}