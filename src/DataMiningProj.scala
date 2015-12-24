import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD

import scala.math.random
import scala.collection.mutable._
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.IDF
import org.apache.spark._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import scala.tools.nsc.io._


object RunIntro {
  var clusters: mllib.clustering.KMeansModel = _;
  var clusterNeighbours: RDD[(linalg.Vector, Array[linalg.Vector])] = _;
  var clusterContents: RDD[(Int, Array[linalg.Vector])] = _;
  var test1: RDD[(linalg.Vector, Array[linalg.Vector])] = _;
  var clusterCentersMap: collection.immutable.Map[Int, linalg.Vector] = _;
  var distinctDocs = 1;

  var step2Total = 0.0;
  var step3Total = 0.0;
  var step4Total = 0.0;
  var totalIterations = 0;
  var inputFile = "";
  var outputFile = "";

  def main(args: Array[String]): Unit = {

    //    var args = Array("docword.test.txt","output.txt")

    // We assumed that the program only gets 2 arguments
    // Input file & output file
    if (args.length != 2) {
      println("Incorrect number of parameters")
      System.exit(-1)
    }

    inputFile = args(0)
    outputFile = args(1)

    val sc = new SparkContext(new SparkConf().setAppName("Intro").setMaster("local"))

    //  val rawblocks = sc.textFile("docword.nips.txt")
    //  def isHeader(line: String) = line.split(' ').length < 2

    File(outputFile).delete()
    val data = sc.textFile(inputFile)
    //  Num of docs
    distinctDocs = data.toArray()(0).toInt
    //  Num of distinct words
    val distinctWords = data.toArray()(1).toInt
    //  Total num of words
    val totalWordCount = data.toArray()(2).toInt

    //  All lines except the first 3
    val cleanData = data.filter(x => x.split(' ').length != 1)

    //  Create a map from docIDs to array of tuples like [ wordID wordFreq,wordID wordFreq ]
    val docToWordTuplesRDD = cleanData.map(line => (line.split(' ')(0).toString, (line.split(' ')(1) + " " + line.split(' ')(2)).toString())).cache()

    val groupedByDoc = docToWordTuplesRDD.groupByKey()

    //  groupedByDoc.foreach(println)
    val docVectors2 = groupedByDoc.map(key => Vectors.sparse(distinctWords + 1, // For each doc
      key._2.toString.replace("CompactBuffer(", "").replace(")", "").split(',').map(x => x.trim().split(' ')(0).toInt), // Get all wordIDs as sparse Vector positions
      key._2.toString.replace("CompactBuffer(", "").replace(")", "").split(',').map(x => x.trim().split(' ')(1).toDouble))) // Get all wordFreqs as sparse Vector contents
    //  docVectors.foreach(println)
    KMeansExecutor(docVectors2)

    var step4StartingTime = System.nanoTime()
    // Maps from a center vector, to all the vectors of its contents
    val clusterCentersToClusterVectors = clusterContents.map(x => (clusterCentersMap.get(x._1), x._2))
    //    clusterCentersToClusterVectors.foreach(x => println(x._2.length))
    // Maps from a cluster center vector, to the closest vector belonging in that cluster
    val centersToClosestClusterVectors = clusterCentersToClusterVectors.map(x => (x._1, getMinPoint(x._1.last, x._2)))
    //    centersToClosestClusterVectors.foreach(x => println(dist(x._1.last,x._2)))
    step4Total = step4Total + (System.nanoTime() - step4StartingTime)

    // Write all array contents to a single string, including line separators
    var representativePoints = ""
    val centersToClosestVectorArray = centersToClosestClusterVectors.collect()
    for (i <- 0 until centersToClosestVectorArray.length) {
      representativePoints = representativePoints + centersToClosestVectorArray(i)._2.toString + System.getProperty("line.separator")
    }

    File(outputFile).appendAll(System.getProperty("line.separator") + "Chosen k: " + clusterCentersMap.keys.size + System.getProperty("line.separator") + representativePoints)

    /*    val hashingTF = new HashingTF()
    val totallyLegitIterable = groupedByDoc.map(x => x._2)*/
    val tf: RDD[linalg.Vector] = docVectors2.cache()
    //val tf: RDD[linalg.Vector] = docVectors.map(x => hashingTF.transform(docVectors.collect().toIterable))
    //      hashingTF.transform(docVectors)

    //    tf.cache()
    val idf = new IDF(minDocFreq = 2).fit(tf)
    val tfidf: RDD[linalg.Vector] = idf.transform(tf)
    //    println("Old vectors: "+docVectors2.count())
    //    docVectors2.foreach(x => println(x.numNonzeros))
    val docVectors = tfidf.cache()
    //    docVectors.foreach(x => println(x.numNonzeros))
    KMeansExecutor(docVectors)

    step4StartingTime = System.nanoTime()
    // Maps from a center vector, to all the vectors of its contents
    val clusterCentersToClusterVectors2 = clusterContents.map(x => (clusterCentersMap.get(x._1), x._2))
    //    clusterCentersToClusterVectors.foreach(x => println(x._2.length))
    // Maps from a cluster center vector, to the closest vector belonging in that cluster
    val centersToClosestClusterVectors2 = clusterCentersToClusterVectors2.map(x => (x._1, getMinPoint(x._1.last, x._2)))
    //    centersToClosestClusterVectors.foreach(x => println(dist(x._1.last,x._2)))
    step4Total = step4Total + (System.nanoTime() - step4StartingTime)


    var representativePoints2 = ""
    val centersToClosestVectorArray2 = centersToClosestClusterVectors2.collect()
    for (i <- 0 until centersToClosestVectorArray2.length) {
      representativePoints2 = representativePoints2 + centersToClosestVectorArray2(i)._2.toString + System.getProperty("line.separator")
    }

    File(outputFile).appendAll(System.getProperty("line.separator") + "Chosen k: " + clusterCentersMap.keys.size + System.getProperty("line.separator") + representativePoints2)

    File(outputFile).appendAll(
      System.getProperty("line.separator") + step2Total / totalIterations / 1000 / 1000 / 1000 + " s" +
        System.getProperty("line.separator") + step3Total / totalIterations / 1000 / 1000 / 1000 + " s" +
        System.getProperty("line.separator") + step4Total / 1000 / 1000 / 1000 / 2 + " s"
    )

    /*   //  Init variables
       var prevError2 = Double.MaxValue
       var err2 = 1.0
       var numClusters2 = 5
       var count2 = 0
       var silhouette2 = -1.0
       var prevSilhouette2 = -1.0
       val upperLimit2 = distinctDocs / 3
       val errorThreshold2 = 0.03
       val silhouetteThreshold2 = -0.03


       println("Initializing complete")
       //   clusters = KMeansModel.load(sc, "kMeansTest100clusters")

   //    while ((Math.abs(err2) > errorThreshold2 && silhouette2 < silhouetteThreshold2) && numClusters2 < upperLimit2) {
         count2 = count2 + 1
         println("Iteration " + count2)
         clusters = KMeans.train(tfidf, numClusters2, 20)
         val WSSE2 = clusters.computeCost(tfidf)
         println("Iteration " + count2)
         println("Within Set Sum of Squared Errors = " + WSSE2)
         err2 = 1.0 - (WSSE2 / prevError2)
         println("Stat: Error: " + err2 + " previous: " + prevError2)
         prevError2 = WSSE2
         silhouette2 = silhouetteWrapper(tfidf)
         println("Stat: Silhouette : " + silhouette2)
         println("Stat: SilhouetteRatio : " + (1 - Math.abs(silhouette2 / prevSilhouette2)))
         prevSilhouette2 = silhouette2
         numClusters2 = numClusters2 + 1 + (Math.abs(silhouette2 - silhouetteThreshold2) * 50).toInt
   //    }

       val whatever2 = clusterContents.map(x => (clusterCentersMap.get(x._1),x._2))
       //val whatever1 = whatever.map(x => (x._1,x._2.map(x1 => (dist(x._1.last,x1),x1))))
       whatever2.foreach(x => println(x._2.length))
       val whatever12 = whatever2.map(x => (x._1,getMinPoint(x._1.last,x._2)))
       whatever12.foreach(x => println(dist(x._1.last,x._2)))*/

    //    tf.cache()
    //    val idf = new IDF(minDocFreq = 2).fit(tf)
    //    val tfidf: RDD[Vector] = idf.transform(tf)

    //clusters.save(sc, "kMeansTest100clustersTEST")

    //  prevError =
    //  while (err > 0.5){
    //    clusters+=1;
    //    kMeans(clusters)
    //    WSSE = kMeans.getWSSE();
    //    err = 1 - (WSSE / prevError)
    //    prevError = WSSE;
    //    println("Error" + err)
    //    }
  }


  def silhouetteWrapper(docVectors: RDD[linalg.Vector]): Double = {
    val clusterCenters = clusters.clusterCenters
    println("Num of cluster centers: " + clusterCenters.size)

    val docToClusterCentersDistance = docVectors.map(x => minDist(x, clusterCenters))
    //    test2.foreach(x => println(x.toArray.deep.mkString("\n")))//

    val backUpClusters = docToClusterCentersDistance.map(x => (x(1), Array(x(2))))
    //    backUpClusters.foreach(x => println(x._1.size))
    //    println("BackupClusters size :"+backUpClusters.keys.toArray().size)

    //    For each cluster, the clusterNeighbours are the centers affected by it
    clusterNeighbours = backUpClusters.distinct().reduceByKey((a, b) => a ++ b)

    //    Convert rdd to centralized array that will be distributed to all workers
    val distinctClusterNeighbours = clusterNeighbours.map(x => (clusters.predict(x._1), x._2.distinct)).collect()

    val rawClusterContents = docVectors.map(x => (clusters.predict(x), Array(x)))

    val test = docVectors.map(x => (x, Array(x)))
    test1 = test.distinct().reduceByKey((a, b) => a ++ b)

    // (Int, Array[Vectors])
    clusterContents = rawClusterContents.reduceByKey((a, b) => a ++ b)

    clusterContents.foreach(x => println(x._2.size))
    clusterContents.foreach(x => println((x._1)))

    clusterCentersMap = clusterCenters.map(x => (clusters.predict(x), x)).toMap

    // It's like the wordcount example
    // Map all silhouette results using the same key, then use reduceByKey, adding values with the same key (which is all of them)
    val silhouetteSum = clusterContents.map(x => (1, silhouette(x, distinctClusterNeighbours) * x._2.length)).reduceByKey((a, b) => a + b).
      first()._2 // We grab the first element of the resulting array (it has only one anyway, because all values had the same key)
    val silhouetteAvg = silhouetteSum / distinctDocs
    return silhouetteAvg
  }

  def silhouette(cluster: (Int, Array[linalg.Vector]), distinctClusterNeighbours: Array[(Int, Array[linalg.Vector])]): Double = {
    var sI = 0.0
    val localClusterNeighbours = distinctClusterNeighbours.filter(_._1 == cluster._1)(0) // Get only the local cluster neighbours by filtering using the first field
    for (i <- 0 until cluster._2.length) {
      val bI = dist(cluster._2(i), // distance of i-th point in this cluster
        minDist(cluster._2(i),
          //            clusters.clusterCenters
          localClusterNeighbours._2
        )(2)) // Last element returned by minDist is the nearest Cluster center (other than its own ofc)
      //println("bI "+bI)
      val aI = avgDist(cluster._2(i), cluster._2)
      //      val aI = dist(cluster._2(i),minDist(cluster._2(i), clusters.clusterCenters)(1))
      //      println("aI "+aI)
      if (aI == -1.0) {
        sI = sI + 1.0
      }
      else {
        if (bI > aI) sI = sI + (bI - aI) / bI
        else sI = sI + (bI - aI) / aI
      }
      //println("sI "+sI)
    }
    if (cluster._2.length == 0) sI = 0.0
    else sI = sI / (cluster._2.length).toDouble
    //    println("Si of cluster "+cluster._1+" with size "+cluster._2.length+" = "+sI)
    return sI
  }

  /*          clusterNeighbours.filter(   // We need to find the clusters affected by this cluster
              {case (key, value) => key == clusters.clusterCenters(cluster._1)} // Match the cluster center vector
                  ).first()._2.distinct  // No problem using first since we're expecting only one element anyway*/
  def avgDist(point: mllib.linalg.Vector, otherPointsInCluster: Array[mllib.linalg.Vector]): Double = {
    var sum = 0.0
    if (otherPointsInCluster.length == 1) return -1.0
    for (i <- 0 until otherPointsInCluster.length) {
      sum = sum + dist(point, otherPointsInCluster(i))
    }
    return sum / (otherPointsInCluster.length - 1).toDouble
  }

  def minDist(point: mllib.linalg.Vector, clusterIDs: Array[mllib.linalg.Vector]): Array[mllib.linalg.Vector] = {
    assert(clusterIDs.length > 0)
    val ownClusterCenter = clusters.clusterCenters(clusters.predict(point))
    //    println("Entered minDist, clusters size "+clusterIDs.length)

    var min = Double.MaxValue
    var minVec = clusterIDs(0)

    for (i <- 0 until clusterIDs.size) {
      if (dist(point, clusterIDs(i)) <= min && (!clusterIDs(i).equals(ownClusterCenter))) {
        min = dist(point, clusterIDs(i))
        minVec = clusterIDs(i)
      }
    }
    //    println("Nearest neighbour found center "+minVec.numNonzeros)
    return Array(point, ownClusterCenter, minVec)
  }

  def dist(a: mllib.linalg.Vector, b: mllib.linalg.Vector): Double = {
    assert(a.size == b.size)
    var sum = 0.0
    for (i <- 0 until a.size) {
      sum = sum + (a(i) - b(i)) * (a(i) - b(i))
    }

    return Math.sqrt(sum)
  }

  def KMeansExecutor(docVectors: RDD[linalg.Vector]) = {
    //  Init variables
    var prevError = Double.MaxValue
    var err = 1.0
    var numClusters = 5
    //    var count = 0
    var silhouette = -1.0
    var prevSilhouette = -1.0
    val upperLimit = distinctDocs / 3
    val errorThreshold = 0.03
    val silhouetteThreshold = -0.03


    println("Initializing complete")
    //   clusters = KMeansModel.load(sc, "kMeansTest100clusters")

    while ((Math.abs(err) > errorThreshold && silhouette < silhouetteThreshold) && numClusters < upperLimit) {
      //      println("Iteration " + count)
      val step2StartingTime = System.nanoTime()
      clusters = KMeans.train(docVectors, numClusters, 20)
      step2Total = step2Total + (System.nanoTime() - step2StartingTime)
      val step3StartingTime = System.nanoTime()
      val WSSE = clusters.computeCost(docVectors)

      //      Print what's going on
      //      println("Within Set Sum of Squared Errors = " + WSSE)
      //      println("Stat: Error: " + err + " previous: " + prevError)
      //      println("Stat: Silhouette : " + silhouette)
      //      println("Stat: SilhouetteRatio : " + (1 - Math.abs(silhouette / prevSilhouette)))
      err = 1.0 - (WSSE / prevError)
      silhouette = silhouetteWrapper(docVectors)
      step3Total = step3Total + (System.nanoTime() - step3StartingTime)

      //      Update previous variables
      totalIterations = totalIterations + 1
      prevError = WSSE
      prevSilhouette = silhouette
      numClusters = numClusters + 1 + (Math.abs(silhouette - silhouetteThreshold) * 50).toInt

      File(outputFile).appendAll(System.getProperty("line.separator") + "WSSE: " + WSSE + System.getProperty("line.separator") + "Silhouette: " + silhouette)

    }
  }

  def getMinPoint(center: linalg.Vector, vectors: Array[linalg.Vector]): linalg.Vector = {
    assert(vectors.length > 0)
    var point = vectors(0)
    for (i <- 0 until vectors.length) {
      if (dist(center, vectors(i)) < dist(center, point))
        point = vectors(i)
    }
    return point
  }
}