/**
 * Created by luke on 02/10/14.
 */
object Scratch extends App{
  case class Example(features:Array[Double], label:Int) {
    override def toString: String = features.mkString("(", ",", ")") + " -> " + label
  }

  val trainingData = Seq(
    Example(Array[Double](0, 0, 0), 0),
    Example(Array[Double](0, 0, 1), 0),
    Example(Array[Double](0, 1, 0), 0),
    Example(Array[Double](1, 0, 0), 0),
    Example(Array[Double](1, 1, 1), 1)
  )

  implicit val features = (0 until 3) map (x => {e:Example => e.features(x)})
  implicit val labels = {e:Example => e.label}

  val tree = DecisionTree(trainingData)
  println(tree.boundary.right)

}
