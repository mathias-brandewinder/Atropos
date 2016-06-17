#I @"../../packages/"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos/Atropos.Regression/bin/Debug/Atropos.Regression.dll"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics
open Atropos.Regression

#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"titanic.csv">
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
