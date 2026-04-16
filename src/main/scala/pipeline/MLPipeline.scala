package pipeline

import domin.{ModelMetrics, ScoredReview}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import pipeline.SentimentPipeline.{FeatureMode, Hybrid, LexiconOnly, TfidfOnly}

object MLPipeline {
  def run(ds: Dataset[ScoredReview])(implicit spark: SparkSession): Unit = {
    val df = SentimentPipeline.prepareText(ds.toDF())
    val Array(train, test) = df.randomSplit(Array(0.8, 0.2), 42L)
    val trainWeighted = addClassWeights(train)
    val tfidfPipeline = new Pipeline()
      .setStages(SentimentPipeline.buildTfidfStages())
      .fit(trainWeighted)
    val trainTfidf = tfidfPipeline.transform(trainWeighted)
    val testTfidf = tfidfPipeline.transform(test)
    trainWeighted.cache()
    trainWeighted.count()
    trainTfidf.cache()
    trainTfidf.count()
    testTfidf.cache()
    testTfidf.count()
    test.cache()
    test.count()
    runLogisticRegression(trainWeighted, test, LexiconOnly, "LR [LexiconOnly]").show()
    runLogisticRegression(trainTfidf, testTfidf, TfidfOnly, "LR [TfidfOnly]").show()
    runLogisticRegression(trainTfidf, testTfidf, Hybrid, "LR [Hybrid]").show()
    runNaiveBayes(trainTfidf, testTfidf).show()
    trainWeighted.unpersist()
    trainTfidf.unpersist()
    testTfidf.unpersist()
    test.unpersist()
  }

  private def addClassWeights(df: DataFrame): DataFrame = {
    val total = df.count().toDouble
    val negCount = df.filter(col("label") === 0.0).count().toDouble
    val posCount = df.filter(col("label") === 1.0).count().toDouble
    df.withColumn("classWeight",
      when(col("label") === 0.0, total / (2 * negCount))
        .otherwise(total / (2 * posCount)))
  }

  private def runLogisticRegression(train: DataFrame, test: DataFrame, mode: FeatureMode, modelName: String)(implicit spark: SparkSession): ModelMetrics = {
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(100)
      .setWeightCol("classWeight")
    val pipeline = new Pipeline().setStages(buildStages(mode, lr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(1.0, 0.1, 0.01, 0.001))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("label").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
    val cvModel = cv.fit(train)
    val lrModel = cvModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages.last
      .asInstanceOf[LogisticRegressionModel]
    printCoefficients(lrModel, mode)
    evalMetrics(cvModel.transform(test), modelName)
  }

  private def runNaiveBayes(train: DataFrame, test: DataFrame)(implicit spark: SparkSession): ModelMetrics = {
    val mode = TfidfOnly
    val nb = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setModelType("complement")
      .setWeightCol("classWeight")
    val pipeline = new Pipeline().setStages(buildStages(mode, nb))
    val paramGrid = new ParamGridBuilder()
      .addGrid(nb.smoothing, Array(0.1, 0.5, 1.0, 2.0))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("label").setMetricName("f1"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
    val cvModel = cv.fit(train)
    val bestNb = cvModel.bestModel
      .asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages.last
      .asInstanceOf[org.apache.spark.ml.classification.NaiveBayesModel]
    println(s"=== [NaiveBayes] Best smoothing: ${bestNb.getSmoothing} ===")
    evalMetrics(cvModel.transform(test), "NaiveBayes [TfidfOnly]")
  }

  private def buildStages(mode: FeatureMode, classifier: PipelineStage): Array[PipelineStage] = {
    mode match {
      case LexiconOnly =>
        val lexiconAssembler = new VectorAssembler()
          .setInputCols(SentimentPipeline.lexiconCols)
          .setOutputCol("lexiconRaw")
        val lexiconScaler = new StandardScaler()
          .setInputCol("lexiconRaw")
          .setOutputCol("features")
          .setWithMean(true)
          .setWithStd(true)
        Array(lexiconAssembler, lexiconScaler, classifier)

      case TfidfOnly =>
        val assembler = new VectorAssembler()
          .setInputCols(SentimentPipeline.featureCols(TfidfOnly))
          .setOutputCol("features")
        Array(assembler, classifier)

      case Hybrid =>
        val lexiconAssembler = new VectorAssembler()
          .setInputCols(SentimentPipeline.lexiconCols)
          .setOutputCol("lexiconRaw")
        val lexiconScaler = new StandardScaler()
          .setInputCol("lexiconRaw")
          .setOutputCol("lexiconScaled")
          .setWithMean(true)
          .setWithStd(true)
        val hybridAssembler = new VectorAssembler()
          .setInputCols(Array("lexiconScaled", SentimentPipeline.tfidfCol))
          .setOutputCol("features")
        Array(lexiconAssembler, lexiconScaler, hybridAssembler, classifier)
    }
  }

  private def printCoefficients(lrModel: LogisticRegressionModel, mode: FeatureMode): Unit = {
    val coeffs = lrModel.coefficients.toArray
    println(s"=== [$mode] Best LR Coefficients ===")
    mode match {
      case LexiconOnly =>
        println(s"Lexicon feature weights: ${coeffs.mkString(", ")}")
      case TfidfOnly =>
        val topCoeffs = coeffs.zipWithIndex
          .filter(_._1 != 0.0)
          .sortBy(x => -math.abs(x._1))
          .take(10)
        println(s"TF-IDF top 10 non-zero weights: ${topCoeffs.map(_._1).mkString(", ")}")
      case Hybrid =>
        println(s"Lexicon feature weights (scaled): ${coeffs.take(4).mkString(", ")}")
        val topTfidf = coeffs.drop(4).zipWithIndex
          .filter(_._1 != 0.0)
          .sortBy(x => -math.abs(x._1))
          .take(10)
        println(s"TF-IDF top 10 non-zero weights: ${topTfidf.map(_._1).mkString(", ")}")
    }
    println(s"Best regParam: ${lrModel.getRegParam}")
    println(s"Best elasticNetParam: ${lrModel.getElasticNetParam}")
    println(s"Non-zero coefficients: ${coeffs.count(_ != 0.0)}")
  }

  private def evalMetrics(predictions: DataFrame, modelName: String)(implicit spark: SparkSession): ModelMetrics = {
    import spark.implicits._
    val predictionAndLabels = predictions
      .select("prediction", "label")
      .as[(Double, Double)]
      .rdd
      .cache()
    val mm = new MulticlassMetrics(predictionAndLabels)
    println("Confusion Matrix:")
    println(mm.confusionMatrix)
    val metrics = ModelMetrics(
      modelName = modelName,
      accuracy = mm.accuracy,
      f1 = mm.weightedFMeasure,
      weightedPrecision = mm.weightedPrecision,
      weightedRecall = mm.weightedRecall
    )
    predictionAndLabels.unpersist()
    metrics
  }
}