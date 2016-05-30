namespace Atropos.Regression

(*
Multi-variate numerical data only implementation of Linear Regression.

There are multiple TODOs listed for each function where it is needed.
*)

[<RequireQualifiedAccess>]
module Linear = 
    open Atropos.Utilities
    open MathNet.Numerics.LinearAlgebra
    type Config = {
        // maximum iterations during learning
        MaxIterations : int
        // if change % falls under that level
        // during learning, exit.
        MinDelta: float
        Alpha: float
        }
    let MSSE (theta:Vector<float>) (y:Vector<float>) (trainingData:Matrix<float>) =
        let m = trainingData.RowCount |> float
        (trainingData * theta)
        |> subtract y
        |> square
        |> divideBy m

    let StepThetas (y:Vector<float>) (initTheta:Vector<float>) (alpha:float) (trainingData:Matrix<float>) =
        let m = trainingData.RowCount |> float
        let err = (trainingData * initTheta)
                  |> subtract y
        let nT = (trainingData * err)
                |> divideVecBy m
                |> multiply alpha
        initTheta |> subtract nT
    //TODO: intelligently choose alpha
    //TODO: intelligently choose initial theta values
    //TODO: use the delta % to break from loop.
    //TODO: integrate into obs/lbl infrastructure (un-wrap and re-wrap as types)
    let TrainModel (actuals:Vector<float>) (config:Config) (trainingData:Matrix<float>) =
        let m = trainingData.RowCount |> float
        let mutable nThetas = DenseVector.zero<float> trainingData.ColumnCount
        let alpha = 0.001
        for i = 0 to config.MaxIterations do
            let r = trainingData
                    |> StepThetas actuals nThetas alpha
            nThetas <- r
        nThetas