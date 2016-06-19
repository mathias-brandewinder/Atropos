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
    
    let definition:FeatureDefinition<Passenger> = [
        "Class 1", fun p -> p.Pclass = 1
        "Class 2", fun p -> p.Pclass = 2
        "Class 3", fun p -> p.Pclass = 3
        ]
    let from = discreteFrom definition

    fun sample ->
        fun pass -> 
            from pass |> Discrete
            //Discrete ([|"1";"2";"3"|], pass.Pclass |> string) 

let ``passenger gender`` : FeatureLearner<Passenger,bool> =
    fun sample ->
        fun pass -> Discrete ([|"male";"female"|], pass.Sex)

let ``adult`` : FeatureLearner<Passenger,bool> =
    [
        "Kid", fun (p:Passenger) -> p.Age < 12.0
        "Adult", fun p -> p.Age >= 12.0
    ]
    |> fromDefinition  

let model = [
    ``passenger age``
    ``passenger class``
    ``passenger gender`` 
    ``adult``
    ]
