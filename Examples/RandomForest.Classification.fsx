(*
Classification example, using AlgLib Random Forest
*)


#load "ClassificationFeatures.fsx"
open ClassificationFeatures

#r @"alglibnet2/lib/alglibnet2.dll"

#load "RandomForest.fsx"
open Atropos.Metrics
open RandomForest

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
    printfn "Pred: %A; Real: %A" (rfClassification o) l
    rfClassification o, l)
|> accuracy
