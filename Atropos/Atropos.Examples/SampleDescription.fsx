// Basic sample statistics

#I @"../../packages/"
#r @"FSharp.Data/lib/net40/FSharp.Data.dll"
open FSharp.Data

type Sample = CsvProvider<"titanic.csv">
type Passenger = Sample.Row

let sample = Sample.GetSample ()

sample.Rows 
|> Seq.length 
|> printfn "Examples: %i"

sample.Rows 
|> Seq.countBy (fun row -> row.Survived)
|> Seq.iter (fun (lbl,cnt) -> printfn "%A: %i" lbl cnt)

// baseline of naive classifier
let mostFrequentClass = 
    sample.Rows 
    |> Seq.countBy (fun row -> row.Survived)
    |> Seq.maxBy snd
    |> fst

let classifier (obs:Sample.Row) = mostFrequentClass

let accuracy = 
    sample.Rows
    |> Seq.averageBy (fun o -> 
        if classifier o = o.Survived
        then 1.
        else 0.)

printfn "Accuracy:  %.2f" accuracy
