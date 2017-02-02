(*
Regression example, using AlgLib Random Forest
*)

#load "Dependencies.fsx"
#load "RegressionFeatures.fsx"
open RegressionFeatures

#load @"RandomForest.fsx"
#I "../packages/"
#r @"alglibnet2/lib/alglibnet2.dll"

open Atropos.Metrics
open RandomForest

let config = { 
    RandomForest.DefaultConfig with 
        Trees = 100
        FeaturesUsed = fun _ -> 2
        ProportionHeldOut = 0.2
    }

let rfRegression = 
    sample
    |> RandomForest.regression config model

sample
|> Seq.map (fun (o,l) -> rfRegression o, l)
|> RMSE
