package pipeline

import domin.ScoredReview
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Dataset

object MLPipeline {
  def run(ds: Dataset[ScoredReview]): Unit = {
    val df = ds.toDF()
    val assembler = new VectorAssembler()
      .setInputCols(Array("sentimentScore", "wordCount", "positiveCount", "negativeCount"))
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(100)

    // TODO
//    val nb = new NaiveBayes()

//    val paramGrid = new ParamGridBuilder()
//      .addGrid(lr.regParam, Array(0.1, 0.01))
//      .build()

    val pipeline = new Pipeline().setStages(Array(assembler, lr))
    val Array(train, test) = df.randomSplit(Array(0.8, 0.2), 42L)
    val model = pipeline.fit(train)
    val results = model.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")
//
//    val cv = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1"))
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(3)

    val auc = evaluator.evaluate(results)
    println(f"AUC-ROC: $auc%.4f")
    results.select("rating", "label", "prediction", "probability").show(10)
  }

}