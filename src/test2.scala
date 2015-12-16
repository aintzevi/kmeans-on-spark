
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD

import scala.math.random
import scala.collection.mutable._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel

object RunIntro {
  var clusters :mllib.clustering.KMeansModel = _;
  var clusterNeighbours : RDD[(linalg.Vector, Array[linalg.Vector])] = _;
  var clusterContents : RDD[(Int, Array[linalg.Vector])] = _;
  var test1 : RDD[(linalg.Vector, Array[linalg.Vector])] = _;
  var clusterCentersMap : collection.immutable.Map[Int,linalg.Vector] = _;
  var distinctDocs = 1;

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Intro").setMaster("local"))
    //  val rawblocks = sc.textFile("docword.nips.txt")
    //  def isHeader(line: String) = line.split(' ').length < 2

    val data = sc.textFile("docword.test.txt")
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
    val docVectors = groupedByDoc.map(key => Vectors.sparse(distinctWords + 1, // For each doc
      key._2.toString.replace("CompactBuffer(", "").replace(")", "").split(',').map(x => x.trim().split(' ')(0).toInt), // Get all wordIDs as sparse Vector positions
      key._2.toString.replace("CompactBuffer(", "").replace(")", "").split(',').map(x => x.trim().split(' ')(1).toDouble))) // Get all wordFreqs as sparse Vector contents
    //  docVectors.foreach(println)


    //  Init variables
    var prevError = Double.MaxValue
    var err = 1.0
    var numClusters = 3
    var count = 0
    var silhouette = -1.0
    var prevSilhouette = -1.0
    val upperLimit = distinctDocs / 3
    val errorThreshold = 0.03
    val silhouetteThreshold = -0.03


    println("Initializing complete")
    //   clusters = KMeansModel.load(sc, "kMeansTest100clusters")

    while ((Math.abs(err) > errorThreshold && silhouette < silhouetteThreshold) && numClusters < upperLimit) {
      count = count + 1
      println("Iteration " + count)
      clusters = KMeans.train(docVectors, numClusters, 20)
      val WSSE = clusters.computeCost(docVectors)
      println("Iteration " + count)
      println("Within Set Sum of Squared Errors = " + WSSE)
      err = 1.0 - (WSSE / prevError)
      println("Stat: Error: " + err + " previous: " + prevError)
      prevError = WSSE
      silhouette = silhouetteWrapper(docVectors)
      println("Stat: Silhouette : " + silhouette)
      println("Stat: SilhouetteRatio : " + (1 - Math.abs(silhouette / prevSilhouette)))
      prevSilhouette = silhouette
      numClusters = numClusters + 1 + (Math.abs(silhouette - silhouetteThreshold) * 50).toInt
    }

    def getMinPoint(center : linalg.Vector, vectors : Array[linalg.Vector]) : linalg.Vector = {
      assert(vectors.length > 0)
      var point = vectors(0)
      for(i<-0 until vectors.length){
        if(dist(center,vectors(i)) < dist(center,point))
          point = vectors(i)
      }
      return point
    }

    val whatever = clusterContents.map(x => (clusterCentersMap.get(x._1),x._2))
    //val whatever1 = whatever.map(x => (x._1,x._2.map(x1 => (dist(x._1.last,x1),x1))))
    whatever.foreach(x => println(x._2.length))
    val whatever1 = whatever.map(x => (x._1,getMinPoint(x._1.last,x._2)))
    whatever1.foreach(x => println(dist(x._1.last,x._2)))

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


  def silhouetteWrapper(docVectors : RDD[linalg.Vector]) : Double = {
    val clusterCenters = clusters.clusterCenters
    println("Num of cluster centers: "+clusterCenters.size)

    val docToClusterCentersDistance = docVectors.map(x => minDist(x, clusterCenters))
    //    test2.foreach(x => println(x.toArray.deep.mkString("\n")))//

    val backUpClusters = docToClusterCentersDistance.map(x => (x(1), Array(x(2))))
    //    backUpClusters.foreach(x => println(x._1.size))
//    println("BackupClusters size :"+backUpClusters.keys.toArray().size)

    //    For each cluster, the clusterNeighbours are the centers affected by it
    clusterNeighbours = backUpClusters.distinct().reduceByKey((a,b) => a ++ b)

    //    Convert rdd to centralized array that will be distributed to all workers
    val distinctClusterNeighbours = clusterNeighbours.map( x=> (clusters.predict(x._1),x._2.distinct)).collect()

    val rawClusterContents = docVectors.map(x => (clusters.predict(x) , Array(x)))

    val test = docVectors.map(x => (x,Array(x)))
    test1 = test.distinct().reduceByKey((a,b) => a++b)

    // (Int, Array[Vectors])
    clusterContents = rawClusterContents.reduceByKey((a,b) => a ++ b)

    clusterContents.foreach(x => println(x._2.size))
    clusterContents.foreach(x => println((x._1)))

    clusterCentersMap = clusterCenters.map(x => (clusters.predict(x),x)).toMap

    // It's like the wordcount example
    // Map all silhouette results using the same key, then use reduceByKey, adding values with the same key (which is all of them)
    val silhouetteSum = clusterContents.map( x => (1,silhouette(x,distinctClusterNeighbours)*x._2.length)).reduceByKey((a,b) => a+b).
      first()._2 // We grab the first element of the resulting array (it has only one anyway, because all values had the same key)
    val silhouetteAvg = silhouetteSum / distinctDocs
    return silhouetteAvg
  }

  def silhouette( cluster: (Int, Array[linalg.Vector]), distinctClusterNeighbours : Array[(Int,Array[linalg.Vector])]) : Double = {
    var sI = 0.0
    val localClusterNeighbours = distinctClusterNeighbours.filter(_._1 == cluster._1)(0) // Get only the local cluster neighbours by filtering using the first field
    for (i<-0 until cluster._2.length){
      val bI = dist(cluster._2(i), // distance of i-th point in this cluster
        minDist(cluster._2(i),
//            clusters.clusterCenters
              localClusterNeighbours._2
                  )(2))    // Last element returned by minDist is the nearest Cluster center (other than its own ofc)
      //println("bI "+bI)
      val aI = avgDist(cluster._2(i),cluster._2)
//      val aI = dist(cluster._2(i),minDist(cluster._2(i), clusters.clusterCenters)(1))
//      println("aI "+aI)
      if (aI == -1.0 ) {
        sI = sI + 1.0
      }
      else {
        if (bI > aI) sI = sI + (bI - aI )/bI
        else sI = sI + (bI - aI )/aI
      }
      //println("sI "+sI)
    }
    if ( cluster._2.length == 0 ) sI=0.0
    else sI = sI/(cluster._2.length).toDouble
    println("Si of cluster "+cluster._1+" with size "+cluster._2.length+" = "+sI)
    return sI
  }

  /*          clusterNeighbours.filter(   // We need to find the clusters affected by this cluster
              {case (key, value) => key == clusters.clusterCenters(cluster._1)} // Match the cluster center vector
                  ).first()._2.distinct  // No problem using first since we're expecting only one element anyway*/
  def avgDist ( point:mllib.linalg.Vector, otherPointsInCluster:Array[mllib.linalg.Vector] ) : Double = {
    var sum = 0.0
    if (otherPointsInCluster.length==1) return -1.0
    for (i<-0 until otherPointsInCluster.length){
      sum = sum + dist(point,otherPointsInCluster(i))
    }
    return sum / (otherPointsInCluster.length-1).toDouble
  }

  def minDist( point:mllib.linalg.Vector, clusterIDs:Array[mllib.linalg.Vector] ) : Array[mllib.linalg.Vector] = {
    assert(clusterIDs.length > 0)
    val ownClusterCenter = clusters.clusterCenters(clusters.predict(point))
//    println("Entered minDist, clusters size "+clusterIDs.length)
    
    var min = Double.MaxValue
    var minVec = clusterIDs(0)
    
    for (i<-0 until clusterIDs.size){
      if(dist(point, clusterIDs(i)) <= min && (!clusterIDs(i).equals(ownClusterCenter)) ){
        min = dist(point, clusterIDs(i))
        minVec = clusterIDs(i)
      }
    }
//    println("Nearest neighbour found center "+minVec.numNonzeros)
    return Array(point, ownClusterCenter, minVec)
  }
    
  def dist( a:mllib.linalg.Vector, b:mllib.linalg.Vector ) : Double = {
    assert(a.size == b.size)
    var sum = 0.0
    for (i<-0 until a.size){
      sum = sum + ( a(i) - b(i) ) * ( a(i) - b(i) )
    }
    
    return Math.sqrt(sum)
  }



  //println(clusterNeighbours.keys.toArray().size)
  //clusterNeighbours.foreach(println)

  //  1) RDD apo docID se tuple [ predict(docID)  minDistFromAllCentersExceptItsOwnClusterCenterID ]
  //
  //  2) RDD backupClusters = clusterCenter se backupClusterCenter
  //
  //  3) backupClusters.reduceByID()
  //
  //
  //
  //
  //
  //
  //
  //



  /* var wordMap = new HashMap[Int,HashMap[Int,Int]]
   val parsedDataList = parsedData.toArray()
   var innerMap = new HashMap[Int,Int]
   var previous = parsedDataList(0).toArray(0).toInt
   for(i<-0 until parsedDataList.length)
   {
       val key = parsedDataList(i).toArray(0).toInt
       val secKey = parsedDataList(i).toArray(1).toInt
       val frequency = parsedDataList(i).toArray(2).toInt
       innerMap.put(secKey,frequency)
       wordMap.put(key,innerMap)
       if(!(previous == key))
       {
          previous = key
          innerMap = new HashMap[Int,Int]
       }
   }
   wordMap.foreach(println)*/

  // Cluster the data into two classes using KMeans
  /*val numClusters = 20
  val numIterations = 20
  //val clusters = KMeansModel.load(sc, "testClusters")

  // Running KMeans algorithm



    var clusterMap = new HashMap[Int,MutableList[mllib.linalg.Vector]]

    var clusterArray = new Array[Int](parsedData.count().toInt)


    val parsedDataList = parsedData.toArray()
    for (i <- 0 until parsedData.count().toInt)
    {

      val j = clusters.predict(parsedDataList(i))
      if(clusterMap.keySet.contains(j))
          clusterMap.put(j,clusterMap.get(j).get+=parsedDataList(i))
      else
        clusterMap.put(j,MutableList(parsedDataList(i)))
    }

//    var i = -1
//    parsedData.foreach( x => clusterArray(i=i.+(1)) = clusters.predict(x))

//    for (i <- 0 until parsedData.count().toInt) {
//      println(clusterArray(i))
//    }


    //parsedData.foreach { x:mllib.linalg.Vector => clusterMap.put(x.size,MutableList(Vectors.dense(1,2,3))) }

//  parsedData.foreach { x => clusterMap.put(clusters.predict(x),clusterMap(clusters.predict(x)).+=(x)) }
//  parsedData.foreach { x => clusterMap(clusters.predict(x)) = clusterMap(clusters.predict(x)).+=(x) }
//  clusters.predict(parsedData).foreach(println)
  //println(clusters.predict(vec))

//  clusterMap.get(0).foreach(println)
  println(clusterMap.keys.size+" "+clusterMap.get(1).size+" "+parsedData.count())

  for (i <- 0 until clusterMap.keySet.size) {
    println(clusterMap.get(i).get.size)
//    println("Item done")
//      println(clusterMap(i))
  }


//  for x in parsedData
//    hashmap +=


  // Evaluate clustering by computing Within Set Sum of Squared Errors
  val WSSSE = clusters.computeCost(parsedData)
  println("Within Set Sum of Squared Errors = " + WSSSE)*/

  /*val noheader = rawblocks.filter(x => !isHeader(x))
  val vecData = rawblocks.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
//  def toDouble(s: String) = {
//   if ("?".equals(s)) Double.NaN else s.toDouble
//  }

  val numClusters = 2
  val numIterations = 20
  val clusters = KMeans.train(vecData, numClusters, numIterations)
  val WSSSE = clusters.computeCost(vecData)
  println("Within Set Sum of Squared Errors = " + WSSSE)

//    def parse(line: String) = {
//      val pieces = line.split(',')
//      val id1 = pieces(0).toInt
//      val id2 = pieces(1).toInt
//      val scores = pieces.slice(2, 11).map(toDouble)
//      val matched = pieces(11).toBoolean
//      MatchData(id1, id2, scores, matched)
//    }
    noheader.foreach(println) */
}

/*
  calculate avg dist from local points
  calculate min dist from external centers


*/