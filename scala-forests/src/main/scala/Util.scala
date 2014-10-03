import scala.math._

/**
 * Created by luke on 03/10/14.
 */
object Util {
  def entropy[T](set:Seq[T])(implicit labels:T=>Int) = {
    def dist = set.view groupBy labels map (_._2.length.toDouble / set.length)
    dist.map(p => -log(p) * p).sum
  }
}
