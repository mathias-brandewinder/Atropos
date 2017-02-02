#load "Dependencies.fsx"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics

#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"Data/titanic.csv">
type Passenger = Sample.Row

let sample = 
    Sample.GetSample().Rows
    |> Seq.map (fun row ->
        row, row.Fare |> float)

let ``passenger age`` : FeatureLearner<Passenger,float> =
    fun sample ->
        fun pass -> pass.Age |> Continuous

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
