﻿#I @"../packages/"
#r @"../Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos.RandomForest/bin/Debug/Atropos.RandomForest.dll"
#r @"../packages/alglibnet2.0.0.0/lib/alglibnet2.dll"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics
open Atropos.RandomForest

#r @"../packages/FSharp.Data.2.3.0/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"titanic.csv">
type Passenger = Sample.Row

// regression model: the goal is to predict
// how much a passenger paid for his/her ticket.
let ``regression example`` () = 

    let fareSample = 
        Sample.GetSample().Rows
        |> Seq.map (fun row ->
            row, row.Fare |> float)

    let ``passenger age`` : FeatureLearner<Passenger,float> =
        fun sample ->
            let avg = avgReplace (sample |> Seq.map (fun (p,_) -> p.Age))
            fun pass -> pass.Age |> avg |> Continuous

    let ``passenger class`` : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> Discrete ([|"1";"2";"3"|], pass.Pclass |> string)

    let ``family travel`` : FeatureLearner<Passenger,float> =
        fun sample ->
            fun pass -> (if pass.Parch > 0 then 1. else 0.) |> Continuous

    let model = [
        ``passenger age``
        ``passenger class``
        ``family travel``
        ]

    let rfRegression = 
        fareSample
        |> RandomForest.regression RandomForest.DefaultConfig model

    fareSample
    |> Seq.map (fun (o,l) -> rfRegression o, l)
    |> RMSE

// classification example: predicting
// whether a passenger survives.
let ``classification example`` () = 

    let survivalSample = 
        Sample.GetSample().Rows
        |> Seq.map (fun row ->
            row, row.Survived)

    let ``passenger age`` : FeatureLearner<Passenger,bool> =
        fun sample ->
            let avg = avgReplace (sample |> Seq.map (fun (p,_) -> p.Age))
            fun pass -> pass.Age |> avg |> Continuous

    let ``passenger class`` : FeatureLearner<Passenger,bool> =
        fun sample ->
            fun pass -> Discrete ([|"1";"2";"3"|], pass.Pclass |> string) 

    let gender : FeatureLearner<Passenger,bool> =
        fun sample ->
            fun pass -> Discrete ([|"male";"female"|], pass.Sex)

    let model = [
        ``passenger age``
        ``passenger class``
        gender
        ]

    let rfClassification = 
        survivalSample
        |> RandomForest.classifier RandomForest.DefaultConfig model 

    survivalSample
    |> Seq.map (fun (o,l) -> 
        rfClassification o, l)
    |> accuracy

``classification example``()
