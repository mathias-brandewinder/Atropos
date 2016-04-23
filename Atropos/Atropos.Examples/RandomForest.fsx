#I @"../../packages/"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos/Atropos.RandomForest/bin/Debug/Atropos.RandomForest.dll"
#r @"alglibnet2/lib/alglibnet2.dll"

open Atropos.Core
open Atropos.RandomForest

#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"titanic.csv">
type Passenger = Sample.Row

let sample = 
    Sample.GetSample().Rows
    |> Seq.map (fun row ->
        row, row.Fare |> float)

let ``passenger age`` : FeatureLearner<Passenger,float> =
    fun sample ->
        fun pass -> pass.Age

let ``passenger class`` : FeatureLearner<Passenger,float> =
    fun sample ->
        fun pass -> pass.Pclass |> float

let ``family travel`` : FeatureLearner<Passenger,float> =
    fun sample ->
        fun pass -> if pass.Parch > 0 then 1. else 0.

let model = [
    ``passenger age``
    ``passenger class``
    ``family travel``
    ]

let featurize =
    model
    |> learnFeatures sample
    |> featurizer

let isNumber = System.Double.IsNaN >> not

// TODO: figure out how to handle missing values
let filtered = 
    sample
    |> Seq.filter (fun (o,l) -> 
        o
        |> featurize
        |> Seq.forall isNumber)
    |> Seq.toList

let rfPredictor = 
    RandomForest.learn model (filtered |> Seq.take 500)

filtered
|> Seq.skip 500
|> Seq.map (fun (o,l) -> rfPredictor o, l)
|> Seq.toList
|> Seq.averageBy (fun (a,b) -> abs (a - b))
