package domin

case class ScoredReview(rating: Double, text: String, sentimentScore: Int, wordCount: Int, positiveCount: Int, negativeCount: Int, label: Double)
