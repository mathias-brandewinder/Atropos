// Logistic example

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

[<RequireQualifiedAccess>]
module Logistic =

    type Config = {
        MaxIterations: int
        Tolerance: float
        }

    let defaultConfig = {
        MaxIterations = 100
        Tolerance = 1.e-4
        } 
    
    // logistic should support 3 cases
    // y = bool, y = int, y = float
    // also potentially weights on each pair
    let train<'Obs,'Lbl> 
        (config:Config)
        (training:('Obs*'Lbl) seq) 
        (features:Features<'Obs>) 
        (label:BinaryLabel<'Lbl>) =

        let input,output =
            training
            |> Seq.map (fun (obs,lbl) -> 
                prepare features obs,
                label.Label lbl
                )
            |> Seq.filter (fun (xs,y) -> 
                xs 
                |> Seq.forall (Option.isSome))
            |> Seq.map (fun (xs,y) -> 
                xs 
                |> Array.map (fun x -> x.Value) 
                |> Seq.collect id 
                |> Seq.toArray,
                y
                )
            |> Seq.toArray
            |> Array.unzip
        
        let learner = IterativeReweightedLeastSquares<LogisticRegression>()
       
        learner.MaxIterations <- config.MaxIterations
        learner.Tolerance <- config.Tolerance

        let regression = learner.Learn (input,output)

        let predict obs =
            let featurized = prepare features obs
            if (isComplete featurized)
            then 
                featurized
                |> Array.map (fun xs -> xs.Value)
                |> Array.collect id 
                |> regression.Decide
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
        (fun (p:Passenger) -> p.Age) |> continuous
        (fun (p:Passenger)  -> p.Embarked) |> categorical ["C";"S";"Q"]
        (fun (p:Passenger)  -> p.Pclass) |> categorical [1;2;3]            
        (fun (p:Passenger)  -> p.Sex) |> categorical ["male";"female"]            
    ]

let labels = (fun (p:Passenger) -> p.Survived) |> BinaryLabel

let predictor, _ = Logistic.train Logistic.defaultConfig training features labels

open Atropos.Measures

sample
|> Seq.map (fun p -> p.Survived, predictor p)
|> accuracy
