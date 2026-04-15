package pipeline

import domin.{Review, ScoredReview}
import org.apache.spark.ml.feature.StopWordsRemover

object SentimentPipeline extends Serializable {
  private def tokenize(text: String): Array[String] = {
//    val remover = new StopWordsRemover()
//      .setInputCol("tokens")
//      .setOutputCol("filteredTokens")
//    StopWordsRemover.loadDefaultStopWords("english").take(10).foreach(println)
    text.toLowerCase
      .replaceAll("[^a-zA-Z\\s]", "")
      .split("\\s+")
      .filter(_.nonEmpty)
  }

  def scoreReview(dict: Map[String, Int])(review: Review): ScoredReview = {
    val tokens = tokenize(review.text)
    val scores = tokens.flatMap(dict.get)
    val pos = scores.count(_ > 0)
    val neg = scores.count(_ < 0)
    val total = scores.sum
    val label = if (review.rating >= 4.0) 1.0 else 0.0
    ScoredReview(review.rating, review.text, total, tokens.length, pos, neg, label)
  }
}