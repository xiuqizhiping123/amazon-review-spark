package pipeline

import domin.{Review, ScoredReview}

object SentimentPipeline extends Serializable {
  private def tokenize(text: String): Array[String] =
    text.toLowerCase
      .replaceAll("[^a-zA-Z\\s]", "")
      .split("\\s+")
      .filter(_.nonEmpty)

  private def scoreTokens(dict: Map[String, Int])(tokens: Array[String]): Int =
    tokens.flatMap(dict.get).sum

  def scoreReview(dict: Map[String, Int])(review: Review): ScoredReview =
    ScoredReview(review.rating, review.text, scoreTokens(dict)(tokenize(review.text)))
}