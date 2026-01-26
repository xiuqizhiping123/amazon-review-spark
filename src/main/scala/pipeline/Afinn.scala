package pipeline

import scala.io.Source

object Afinn {
  def load(): Map[String, Int] =
    Source.fromInputStream(getClass.getResourceAsStream("/afinn-111.txt"))
      .getLines()
      .map(_.split("\t"))
      .collect { case Array(word, score) => word -> score.toInt }
      .toMap
}