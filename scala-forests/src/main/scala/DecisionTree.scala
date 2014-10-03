import scala.util.Random

/**
 * Created by luke on 02/10/14.
 */
import math._
import Util._

case class DecisionBoundary[T](examples:Seq[T], feature:T=>Double)(val threshold:Double) {
  private val partition = examples.partition(feature(_)<threshold)
  val left = partition._1
  val right = partition._2
}

case class DecisionTree[T](examples:Seq[T])(implicit features:Seq[T=>Double], implicit val labels:T=>Int) {
  private val f = features(Random.nextInt(features.length))
  private val randomExamples = Seq.fill(10)( examples(Random.nextInt(examples.length)) )
  private val randomBoundaries = randomExamples.view map f map DecisionBoundary(examples, f)
  val boundary = randomBoundaries.minBy( x => abs(x.left.length - x.right.length))

  val leftTree = if(entropy(boundary.left) > 0.1) Some(DecisionTree(boundary.left)) else None
  val rightTree = if(entropy(boundary.right) > 0.1) Some(DecisionTree(boundary.right)) else None
}