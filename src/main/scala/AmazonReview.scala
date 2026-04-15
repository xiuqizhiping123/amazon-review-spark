import domin.Review
import io.DatasetLoader
import org.apache.spark.sql.SparkSession
import pipeline.{Afinn, MLPipeline, SentimentPipeline}

object AmazonReview {
  private val datasetPath = "data/amazon-review/All_Beauty.jsonl"
  private val datasetUrl = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Amazon Review Sentiment Analysis")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    DatasetLoader.ensureDatasetAvailable(datasetUrl, datasetPath)

    val reviews = spark.read.json(datasetPath)
      .select("rating", "text", "title")
      .flatMap(Review.fromRow)

    val dict = Afinn.load()
    val dictBc = spark.sparkContext.broadcast(dict)
    val scored = reviews.map(SentimentPipeline.scoreReview(dictBc.value))

    scored.show(5)
    scored.groupBy("rating").count().orderBy("rating").show()

    MLPipeline.run(scored)
  }
}