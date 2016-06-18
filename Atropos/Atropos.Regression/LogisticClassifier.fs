namespace Atropos.Regression

[<RequireQualifiedAccess>]
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
