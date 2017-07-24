namespace Atropos

module Sampling = 

    open System

    // Fisher-Yates shuffle
    // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm
    let shuffle (rng:Random) (xs:_[]) =
        let ys = Array.copy xs
        for i in 0 .. (xs.Length - 1) do
            let j = rng.Next(0, i + 1)
            if i <> j then 
                ys.[i] <- ys.[j]
            ys.[j] <- xs.[i]
        ys

module Core = 

    /// A Feature can be either:
    /// Continuous: it can take any float value,
    /// Categorical: it is one of a set of possible cases.
    type Feature =
        | Continuous  of float
        | Categorical of int * int

    type Features<'Obs> = ('Obs -> Feature) seq
    
    type BinaryLabel<'Lbl> (lbl:'Lbl->bool) =
        member this.Label (x:'Lbl) = lbl x

    type ContinuousLabel<'Lbl> (lbl:'Lbl->float) =
        member this.Label (x:'Lbl) = lbl x

    let caseMatch matches value =
        let cases = matches |> Seq.length
        let index = 
            matches
            |> Seq.findIndex (fun m -> m = value)
        (index,cases)

    let properNumber (x:float) =
        if System.Double.IsInfinity x
        then None
        elif System.Double.IsNaN x
        then None
        else Some x    

    let inline continuous x = float x |> Continuous

    let categorical matches = caseMatch matches >> Categorical

    let encode<'Obs> (feature:'Obs -> Feature) (obs:'Obs) =
        try
            let value = feature obs
            match value with
            | Continuous(v) -> 
                properNumber v
                |> Option.map (Seq.singleton)
            | Categorical(index,cases) ->
                Seq.init (cases - 1) (fun i -> if i = index then 1. else 0.)
                |> Some
        with
        | _ -> None 

    let prepare (features:Features<'Obs>) (obs:'Obs) =
        features
        |> Seq.map (fun feature -> 
            encode feature obs 
            |> Option.map (Seq.toArray))
        |> Seq.toArray

    let available = Option.isSome
    let isComplete<'T> (xs:Option<'T> seq) = 
        xs |> Seq.forall (available)

