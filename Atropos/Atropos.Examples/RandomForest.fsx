#I @"../../packages/"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos/Atropos.RandomForest/bin/Debug/Atropos.RandomForest.dll"
#r @"alglibnet2/lib/alglibnet2.dll"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics
// TODO: should this be Atropos.Alglib.RandomForest?
// TODO: should this be a separate package?
open Atropos.RandomForest

#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
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
        RandomForest.regression model fareSample

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

    // this should be simplified
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
        |> RandomForest.classifier model 

    survivalSample
    |> Seq.map (fun (o,l) -> 
        rfClassification o, l)
    |> Seq.averageBy (fun (a,b) -> if a = b then 1. else 0.)
