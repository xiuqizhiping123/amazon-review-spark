package io

import java.io.{File, FileOutputStream}
import java.net.{HttpURLConnection, URL}
import scala.util.{Failure, Success, Try, Using}

object DatasetLoader {
  def ensureDatasetAvailable(url: String, path: String): Unit = {
    if (datasetExists(path)) {
      println(s"Dataset already exists at $path. Skipping download.")
    } else {
      println(s"Dataset not found. Starting download from: $url")
      downloadDataset(url, path) match {
        case Success(_) => println("Dataset downloaded successfully.")
        case Failure(e) => throw new RuntimeException(s"Download failed: ${e.getMessage}")
      }
    }
  }

  private def datasetExists(path: String): Boolean = {
    val file = new File(path)
    file.exists() && file.isFile && file.length() > 0
  }

  private def downloadDataset(url: String, savePath: String): Try[Unit] = Try {
    Option(new File(savePath).getParentFile).foreach(_.mkdirs())
    val connection = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    connection.setRequestMethod("GET")
    require(
      connection.getResponseCode == HttpURLConnection.HTTP_OK,
      s"Server returned: ${connection.getResponseCode}"
    )
    try {
      Using.Manager { use =>
        val in = use(connection.getInputStream)
        val out = use(new FileOutputStream(savePath))
        Iterator.continually {
            val data = new Array[Byte](8192)
            (in.read(data), data)
          }.takeWhile(_._1 != -1)
          .foreach { case (read, data) => out.write(data, 0, read) }
      }.get
    } finally {
      connection.disconnect()
    }
  }

}
