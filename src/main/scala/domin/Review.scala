package domin

import org.apache.spark.sql.Row

case class Review(rating: Double, text: String, title: String)

object Review {
  def fromRow(row: Row): Option[Review] =
    for {
      rating <- Option(row.getAs[Double]("rating"))
      text <- Option(row.getAs[String]("text")).filter(_.nonEmpty)
      title = Option(row.getAs[String]("title")).getOrElse("")
    } yield Review(rating, text, title)
}
