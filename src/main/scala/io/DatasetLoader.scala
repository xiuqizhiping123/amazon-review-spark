package io

import java.io.{File, FileOutputStream}
import java.net.{HttpURLConnection, URL}
import scala.util.{Failure, Success, Try}

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
    val dir = new File(savePath).getParentFile
    if (dir != null && !dir.exists()) dir.mkdirs()
    val connection = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    connection.setRequestMethod("GET")
    val responseCode = connection.getResponseCode
    if (responseCode != HttpURLConnection.HTTP_OK)
      throw new RuntimeException(s"Response Code: $responseCode")
    val inputStream = connection.getInputStream
    val outputStream = new FileOutputStream(savePath)
    try {
      val buffer = new Array[Byte](8192)
      var totalRead = 0L
      var bytesRead = inputStream.read(buffer)
      while (bytesRead != -1) {
        outputStream.write(buffer, 0, bytesRead)
        totalRead += bytesRead
        bytesRead = inputStream.read(buffer)
      }
    } finally {
      inputStream.close()
      outputStream.close()
      connection.disconnect()
    }
  }

}
