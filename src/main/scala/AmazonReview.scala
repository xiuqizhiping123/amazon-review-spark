import domin.Review
import io.DatasetLoader
import org.apache.spark.sql.SparkSession
import pipeline.{Afinn, MLPipeline, SentimentPipeline}

object AmazonReview {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession.builder
      .appName("Amazon Review Sentiment Analysis")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val categories = Seq(
      "All_Beauty" -> "data/amazon-review/All_Beauty.jsonl",
      "Gift_Cards" -> "data/amazon-review/Gift_Cards.jsonl",
    )

    val baseUrl = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories"

    val allResults = categories.flatMap { case (name, path) =>
      println(s"Category: $name")
      DatasetLoader.ensureDatasetAvailable(s"$baseUrl/$name.jsonl", path)

      val reviews = spark.read.json(path)
        .select("rating", "text", "title")
        .flatMap(Review.fromRow)

      val dict = Afinn.load()
      val dictBc = spark.sparkContext.broadcast(dict)
      val scored = reviews.map(SentimentPipeline.scoreReview(dictBc.value))

      scored.groupBy("rating").count().orderBy("rating").show()
      MLPipeline.run(scored, name)
    }

    // Export as a single partitioned CSV
    allResults.toDS().coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("data/results/metrics")
  }
}