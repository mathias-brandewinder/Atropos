#I @"../../packages/"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics

#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"titanic.csv">
type Passenger = Sample.Row

let sample = 
    Sample.GetSample().Rows
    |> Seq.map (fun row ->
        row, row.Survived)

let ``passenger age`` : FeatureLearner<Passenger,bool> =
    fun sample ->
        fun pass -> pass.Age |> Continuous

let ``passenger class`` : FeatureLearner<Passenger,bool> =
    fun sample ->
        fun pass -> Discrete ([|"1";"2";"3"|], pass.Pclass |> string) 

let ``passenger gender`` : FeatureLearner<Passenger,bool> =
    fun sample ->
        fun pass -> Discrete ([|"male";"female"|], pass.Sex)

let model = [
    ``passenger age``
    ``passenger class``
    ``passenger gender`` 
    ]
