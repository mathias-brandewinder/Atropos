#I @"../../packages/"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos/Atropos.RandomForest/bin/Debug/Atropos.RandomForest.dll"
#r @"alglibnet2/lib/alglibnet2.dll"

open Atropos.Core
open Atropos.Utilities
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
        let avg = avgReplace (sample |> Seq.map (fun (p,_) -> p.Age))
        fun pass -> pass.Age |> avg

let ``passenger class`` : FeatureLearner<Passenger,float> =
    fun sample ->
        let avg = avgReplace (sample |> Seq.map (fun (p,_) -> p.Pclass |> float))
        fun pass -> pass.Pclass |> float |> avg

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

// TODO: figure out how to handle missing values
// 2 possible strategies:
// 1) value replacement
// 2) filter out incomplete observations

// perhaps a reasonable strategy is
// "define feature behavior if you want, but
// filter out unusable rows anyways"

let rfPredictor = 
    RandomForest.learn model (sample |> Seq.take 500)

sample
|> Seq.skip 500
|> Seq.map (fun (o,l) -> rfPredictor o, l)
|> Seq.toList
|> Seq.averageBy (fun (a,b) -> abs (a - b))
