(*
Classification example, using Accord Support Vector Machine
*)

#load "ClassificationFeatures.fsx"
open ClassificationFeatures

#I @"../../packages/"
#r @"Accord.Statistics/lib/net45/Accord.Statistics.dll"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos/Atropos.SVM/bin/Debug/Atropos.SVM.dll"

open Atropos.Metrics
open Atropos.SVM
open Accord.Statistics.Kernels

let config = { 
    SVM.DefaultConfig with 
        Tolerance = 1e-3
        Kernel = Gaussian ()
        Complexity = Some 5.0 }

let svmClassifier = 
    sample
    |> SVM.classification config model 

// probably an issue here: what is predicted
// when there is an issue with featurization,
// i.e. the vector contains NaNs?
// TODO figure out how to handle.
sample
|> Seq.map (fun (o,l) -> 
    svmClassifier o, l)
|> Seq.map (fun (p,l) -> 
    printfn "Pred: %b, Act: %b" p l
    p,l)
|> accuracy
