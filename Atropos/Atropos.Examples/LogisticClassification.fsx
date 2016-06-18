(*
Classification example, using Accord Logistic Regression
*)

#load "ClassificationFeatures.fsx"
open ClassificationFeatures

#I @"../../packages/"
#r @"Accord.Statistics/lib/net45/Accord.Statistics.dll"
#r @"../Atropos/Atropos.Regression/bin/Debug/Atropos.Regression.dll"

open Atropos.Regression
open Atropos.Metrics

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
