/**
 * this object is to train and save model
 */
object Main extends App{
  /**
   * run 3 model and save the model
   */
  val dt = DT.Run()
  val rf2 = RF2.Run()
  val lr = LR.Run()
}
