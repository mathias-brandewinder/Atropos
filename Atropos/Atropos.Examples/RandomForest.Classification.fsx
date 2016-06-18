(*
Classification example, using AlgLib Random Forest
*)

#load "ClassificationFeatures.fsx"
open ClassificationFeatures

#r @"../Atropos/Atropos.RandomForest/bin/Debug/Atropos.RandomForest.dll"
#r @"alglibnet2/lib/alglibnet2.dll"

open Atropos.Metrics
open Atropos.RandomForest

let config = { 
    RandomForest.DefaultConfig with 
        Trees = 1000
        FeaturesUsed = fun _ -> 3
        ProportionHeldOut = 0.2
    }

let rfClassification = 
    sample
    |> RandomForest.classifier config model 

sample
|> Seq.map (fun (o,l) -> 
    rfClassification o, l)
|> accuracy
