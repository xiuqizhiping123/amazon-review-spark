package domin

case class ModelMetrics(category: String = "", modelName: String, accuracy: Double, f1: Double, weightedPrecision: Double,
                        weightedRecall: Double, tn: Long, fp: Long, fn: Long, tp: Long) {
  private def class0Recall: Double = tn.toDouble / (tn + fp)

  private def class1Recall: Double = tp.toDouble / (fn + tp)

  def show(): Unit = {
    println(
      f"""
         |=== $modelName ($category) ===
         |Accuracy:         $accuracy%.4f
         |F1 Score:         $f1%.4f
         |Precision:        $weightedPrecision%.4f
         |Recall:           $weightedRecall%.4f
         |Class 0 Recall:   $class0Recall%.4f
         |Class 1 Recall:   $class1Recall%.4f
         |Confusion Matrix:
         |  TN=$tn  FP=$fp
         |  FN=$fn  TP=$tp
         """.stripMargin)
  }
}