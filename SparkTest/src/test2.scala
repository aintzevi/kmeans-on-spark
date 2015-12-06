
import scala.math.random
import scala.collection.mutable._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeans

object RunIntro extends Serializable {
  def main(args: Array[String]): Unit = {
  val sc = new SparkContext(new SparkConf().setAppName("Intro").setMaster("local"))
   
//  val rawblocks = sc.textFile("docword.nips.txt")
//  def isHeader(line: String) = line.split(' ').length < 2
  
  
  val data = sc.textFile("docword.txt")
  val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

  // Cluster the data into two classes using KMeans
  val numClusters = 610
  val numIterations = 17
  val clusters = KMeansModel.load(sc, "testClusters")
  
  // Running KMeans algorithm 
  
//  val clusters = KMeans.train(parsedData, numClusters, numIterations)
//  clusters.save(sc, "testClusters")
  
    var clusterMap = new HashMap[Int,MutableList[mllib.linalg.Vector]]
    
    
//  val rddMap = sc.parallelize(Map().toSeq)
  
  //var clusterMap:Map[Int,MutableList[mllib.linalg.Vector]] = Map()
//  val rdd = sc.parallelize(clusterMap.toSeq)
  
//  clusters.clusterCenters.foreach(println)
//  val vec = Vectors.dense(-1,-1,-1)
    var clusterArray = new Array[Int](parsedData.count().toInt)

//    clusterMap.put(0,MutableList(Vectors.dense(4,5,6)))
    val parsedDataList = parsedData.toArray()
    for (i <- 0 until parsedData.count().toInt) 
    {
//      clusterArray(i) = clusters.predict(parsedDataList(i))
//      if (clusterMap())
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
  println("Within Set Sum of Squared Errors = " + WSSSE)
  
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
 }