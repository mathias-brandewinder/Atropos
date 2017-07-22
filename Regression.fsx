// Linear regression example

#load "Atropos.fs"
open Atropos.Core

#I "./packages/"
#r "Accord/lib/net45/Accord.dll"
#r "Accord.Math/lib/net45/Accord.Math.dll"
#r "Accord.Math/lib/net45/Accord.Math.Core.dll"
#r "Accord.Statistics/lib/net45/Accord.Statistics.dll"
#r "Accord.MachineLearning/lib/net45/Accord.MachineLearning.dll"

open Accord.Statistics
open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting

// potentially weights on each pair
let linear<'Obs,'Lbl> 
    (training: ('Obs * 'Lbl) seq) 
    (features: Features<'Obs>) 
    (label: ContinuousLabel<'Lbl>) =

    let input,output =
        training
        |> Seq.map (fun (obs,lbl) -> 
            prepare features obs,
            label.Label lbl
            )
        |> Seq.filter (fun (xs,y) -> 
            xs |> Seq.forall (fun x -> x |> Option.isSome))
        |> Seq.map (fun (xs,y) -> 
            xs |> Array.map (fun x -> x.Value) |> Seq.collect id |> Seq.toArray, y)
        |> Seq.toArray
        |> Array.unzip
    
    let learner = Linear.OrdinaryLeastSquares()
    // how to handle specialized params like that?
    learner.IsRobust <- true
    
    let regression = learner.Learn (input,output)

    let predict obs =
        let featurized = prepare features obs
        if (isComplete featurized)
        then 
            featurized
            |> Array.map (fun xs -> xs.Value)
            |> Array.collect id 
            |> regression.Transform
            |> Some
        else None

    predict, regression

// test on Titanic

#I "./packages/"
#r "FSharp.Data/lib/net40/FSharp.Data.dll"

open FSharp.Data

type Titanic = CsvProvider<"titanic.csv">
type Passenger = Titanic.Row

let sample = Titanic.GetSample().Rows
let training = sample |> Seq.map (fun x -> x,x)

let features = 
    [
        (fun (p:Passenger) -> p.Age) >> continuous
        (fun p -> p.Embarked) >> categorical ["C";"S";"Q"]
        (fun p -> p.Pclass) >> categorical [1;2;3]            
        (fun p -> p.Sex) >> categorical ["male";"female"]            
    ]

let labels = (fun (p:Passenger) -> p.Fare |> float) |> ContinuousLabel

let predictor, _ = linear training features labels

sample
|> Seq.map (fun p -> 
    predictor p 
    |> Option.map (fun x -> abs (x - float p.Fare))
    )
|> Seq.choose id
|> Seq.average

sample
|> Seq.take 50
|> Seq.iter (fun p -> 
    printfn "%A,%A" p.Fare (predictor p) 
    )
