package pipeline

import domin.ScoredReview
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Dataset

object MLPipeline {

  def assemble(ds: Dataset[ScoredReview]) = {
    val assembler = new VectorAssembler()
      .setInputCols(Array("sentimentScore", "wordCount", "positiveCount", "negativeCount"))
      .setOutputCol("features")

    assembler.transform(ds.toDF())
  }
}