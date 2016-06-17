namespace Atropos.Regression

(*
TODO
- handle different label options
    - discrete: is it binary
    - continuous: is it in [0;1]
    - if continuous, should I normalize if I have bounds information?
    - 'positive/negative per case'
    - how about multi-class, multi-label?
*)

[<RequireQualifiedAccess>]
module Logistic = 

    open Atropos.Utilities
    open Atropos.Core
    open Accord.Statistics.Models.Regression
    open Accord.Statistics.Models.Regression.Fitting

    type Config = {
        // maximum iterations during learning
        MaxIterations : int
        // if change % falls under that level
        // during learning, exit.
        MinDelta: float
        }

    let DefaultConfig = {
        MaxIterations = 100 
        MinDelta = 0.001
        }

    let regression : Config -> Learner<'Obs,float> =

        fun config ->
            fun model ->
                fun sample ->

                    let featurize =
                        model
                        |> learnFeatures sample
                        |> featurizer

                    let features, labels =
                        sample
                        |> Seq.map (fun (obs,lbl) ->
                            let fs = featurize obs
                            fs,lbl)
                        // discard any row with missing data
                        |> Seq.filter (fun (obs,lbl) -> numbers obs && number lbl)
                        // TODO: can I have one-shot Array extraction?
                        |> Seq.toArray
                        |> Array.unzip

                    let featuresCount = features.[0].Length

                    let logisticReg = LogisticRegression(featuresCount)
                    let learner = LogisticGradientDescent(logisticReg)
                    let sampleSize = labels.Length |> float

                    // TODO: collect metrics of interest during learning?
                    let rec improve iteration =
                        // TODO confirm what delta is: currently the 
                        // delta-based termination rule causes weird results.
                        let delta = learner.Run(features,labels) // sampleSize
//                        printfn "Delta: %.3f" delta
//                        if delta < config.MinDelta
//                        then ignore ()
                        if iteration > config.MaxIterations
                        then ignore ()
                        else improve (iteration + 1)
                
                    improve 0
                
                    let predictor = 
                        featurize >> logisticReg.Compute

                    predictor
