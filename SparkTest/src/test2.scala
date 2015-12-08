
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
  //val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
 val keyValueRDD = data.map(line => (line.split(' ')(0).toString,(line.split(' ')(1)+" "+line.split(' ')(2)).toString())).cache()
  val groupedByKey = keyValueRDD.groupByKey()
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
 }