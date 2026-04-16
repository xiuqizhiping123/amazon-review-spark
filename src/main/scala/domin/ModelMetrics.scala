package domin

case class ModelMetrics(modelName: String, accuracy: Double, f1: Double, weightedPrecision: Double, weightedRecall: Double) {
  def show(): Unit = println(
    f"""
       |=== $modelName ===
       |Accuracy:   $accuracy%.4f
       |F1 Score:   $f1%.4f
       |Precision:  $weightedPrecision%.4f
       |Recall:     $weightedRecall%.4f
       """.stripMargin)
}