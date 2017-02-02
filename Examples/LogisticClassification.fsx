(*
Classification example, using Accord Logistic Regression
*)

// TODO: CURRENTLY BROKEN, NEEDS FIXING

#load "Dependencies.fsx"
open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics

#I @"../packages/"
#r @"Accord/lib/net46/Accord.dll"
#r @"Accord.Math/lib/net46/Accord.Math.dll"
#r @"Accord.Statistics/lib/net46/Accord.Statistics.dll"
open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting

module LogisticClassifier =

    open Atropos.Utilities
    open Atropos.Core
    open Accord.Statistics.Models.Regression
    open Accord.Statistics.Models.Regression.Fitting

    type Config = {
        // maximum iterations during learning
        MaxIterations : int
        }

    let DefaultConfig = {
        MaxIterations = 100 
        }

    let regression : Config -> Learner<'Obs,'Lbl> =

        fun config ->
            fun model ->
                fun sample ->

                    let featurize = featuresExtractor model sample

                    let cases =
                        sample
                        |> Seq.map (fun (_,lbl) -> lbl)
                        |> Seq.distinct
                        |> Seq.toArray

                    // Logistic should be binary.
                    if cases.Length <> 2 then failwith "should be 2 cases"

                    let normalize label =
                        cases
                        |> Array.findIndex (fun case -> case = label)
                        |> float

                    let denormalize proba =
                        let index = 
                            if proba > 0.5 then 1 else 0
                        cases.[index]

                    let features, labels =
                        sample
                        |> Seq.map (fun (obs,lbl) ->
                            let fs = featurize obs
                            let l = normalize lbl
                            fs,l)
                        // discard any row with missing data
                        |> Seq.filter (fun (obs,_) -> numbers obs)
                        // TODO: can I have one-shot Array extraction?
                        |> Seq.toArray
                        |> Array.unzip

                    let featuresCount = features.[0].Length

                    let logisticReg = LogisticRegression(featuresCount)
                    let learner = LogisticGradientDescent(logisticReg)
                    let sampleSize = labels.Length |> float

                    let rec improve iteration =
                        let _ = learner.Run(features,labels)
                        if iteration > config.MaxIterations
                        then ignore ()
                        else improve (iteration + 1)
                
                    improve 0
                
                    let predictor = 
                        featurize >> logisticReg.Compute >> denormalize

                    predictor


#load "ClassificationFeatures.fsx"
open ClassificationFeatures

let config = { 
    LogisticClassifier.DefaultConfig with 
        MaxIterations = 10000 }

let logRegression = 
    sample
    |> LogisticClassifier.regression config model 

sample
|> Seq.iter (fun (o,l) -> 
    printfn "Pred: %b Real: %b" (logRegression o) l)
    
let accuracy = 
    sample
    |> Seq.map (fun (o,l) -> 
        logRegression o, l)
    |> accuracy

printfn "Accuracy:  %.2f" accuracy
