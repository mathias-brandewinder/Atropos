#load "Dependencies.fsx"
open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics

#r @"Accord/lib/net46/Accord.dll"
#r @"Accord.Math/lib/net46/Accord.Math.dll"
#r @"Accord.Statistics/lib/net46/Accord.Statistics.dll"
open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting

(*
TODO
- handle different label options
    - discrete: is it binary
    - continuous: is it in [0;1]
    - if continuous, should I normalize if I have bounds information?
    - 'positive/negative per case'
    - how about multi-class, multi-label?
*)

module Logistic = 

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

                    let featurize = featuresExtractor model sample

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


#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"Data/titanic.csv">
type Passenger = Sample.Row

let ``classification model`` () = 

    let survivalSample = 
        Sample.GetSample().Rows
        |> Seq.map (fun row ->
            row, if row.Survived then 1. else 0.)

    // this should be simplified
    let ``passenger age`` : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> pass.Age |> Continuous

    let ``passenger class`` : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> Discrete ([|"1";"2";"3"|], pass.Pclass |> string) 

    let ``passenger gender`` : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> Discrete ([|"male";"female"|], pass.Sex)

    let model = [
        ``passenger age``
        ``passenger class``
        ``passenger gender``
        ]

    let config = { Logistic.DefaultConfig with MaxIterations = 10000 }
    let logRegression = 
        survivalSample
        |> Logistic.regression config model 

    survivalSample
    |> Seq.iter (fun (o,l) -> 
        printfn "Pred: %.3f Real: %.1f" (logRegression o) l)
    
    let rmse = 
        survivalSample
        |> Seq.map (fun (o,l) -> 
            logRegression o, l)
        |> Seq.filter (fun (p,l) -> number p)
        |> RMSE

    let accuracy = 
        survivalSample
        |> Seq.map (fun (o,l) -> 
            logRegression o, l)
        |> Seq.filter (fun (p,l) -> number p)
        // TODO figure out what to do there! this is nasty.
        |> Seq.map (fun (p,l) -> (if p > 0.5 then 1. else 0.),l)
        |> accuracy

    printfn "RMSE:      %.2f" rmse
    printfn "Accuracy:  %.2f" accuracy

``classification model`` ()
