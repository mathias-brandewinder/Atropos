#I @"../../packages/"
#r @"Accord.Statistics/lib/net45/Accord.Statistics.dll"
#r @"../Atropos/Atropos/bin/Debug/Atropos.dll"
#r @"../Atropos/Atropos.SVM/bin/Debug/Atropos.SVM.dll"

open Atropos.Core
open Atropos.Utilities
open Atropos.Metrics
open Atropos.SVM
open Accord.Statistics.Kernels

#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"titanic.csv">
type Passenger = Sample.Row

let ``classification model`` () = 

    let survivalSample = 
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

    let config = { 
        SVM.DefaultConfig with 
            Tolerance = 1e-3
            Kernel = Gaussian ()
            Complexity = Some 5.0 }

    let svmClassifier = 
        survivalSample
        |> SVM.classification config model 

    survivalSample
    // probably an issue here: what is predicted
    // when there is an issue with featurization,
    // i.e. the vector contains NaNs?
    // TODO figure out how to handle.
    |> Seq.map (fun (o,l) -> 
        svmClassifier o, l)
    |> Seq.map (fun (p,l) -> 
        printfn "Pred: %b, Act: %b" p l
        p,l)
    |> accuracy

``classification model`` ()