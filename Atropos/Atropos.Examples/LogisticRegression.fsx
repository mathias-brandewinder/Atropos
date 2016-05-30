﻿#I @"../packages/"
#r @"../Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos.Regression/bin/Debug/Atropos.Regression.dll"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics
open Atropos.Regression

#r @"../packages/FSharp.Data.2.3.0/lib/net40/FSharp.Data.dll"
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
            let avg = avgReplace (sample |> Seq.map (fun (p,_) -> p.Age))
            fun pass -> pass.Age |> avg |> Continuous

    let ``passenger class`` : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> Discrete ([|"1";"2";"3"|], pass.Pclass |> string) 

    let gender : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> Discrete ([|"male";"female"|], pass.Sex)

    let model = [
        ``passenger age``
        ``passenger class``
        gender
        ]

    let logRegression = 
        survivalSample
        |> Logistic.regression Logistic.DefaultConfig model 

    survivalSample
    |> Seq.map (fun (o,l) -> 
        logRegression o, l)
    |> RMSE