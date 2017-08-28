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

    /// A Value extracted from an Observation can either
    /// be Valid, or Invalid, in what case we provide both
    /// the faulty observation, and an error message.
    type Value<'Obs,'T> =
        | Valid of 'T
        | Invalid of 'Obs * string

    let isValid = function 
        | Valid(_) -> true 
        | Invalid(_) -> false

    let extractValid xs =
        xs
        |> Seq.filter isValid
        |> Seq.map (function 
            | Valid(x) -> x 
            | Invalid(_) -> failwith "Impossible"
            )

    /// A Feature can be either:
    /// Continuous: it can take any float value,
    /// Categorical: it is one of a set of possible cases.
    type Feature<'Obs> = 
        | Continuous of ('Obs -> Value<'Obs,float>)
        | Categorical of (int * ('Obs -> Value<'Obs,int>))
    
    type BinaryLabel<'Lbl> (lbl:'Lbl->bool) =
        member this.Label (x:'Lbl) = lbl x

    type ContinuousLabel<'Lbl> (lbl:'Lbl->float) =
        member this.Label (x:'Lbl) = lbl x

    /// Given a feature, extracting a value from an observation,
    /// and a list of expected matching values, return both
    /// the number of possible cases, and the 0-based index
    /// indicating which case is active.
    let categorical matches feat =
        let cases = matches |> Seq.length
        let activeCase = 
            fun obs ->
                try
                    let value = feat obs
                    let index = 
                        matches
                        |> Seq.tryFindIndex (fun m -> m = value)
                    match index with
                    | Some(index) -> Valid(index)
                    | None -> 
                        let msg = sprintf "Unexpected value: %A" value
                        Invalid(obs,msg)
                with
                | ex -> Invalid(obs,ex.Message)

        Categorical(cases, activeCase)

    let inline continuous feat =
        let extractor = 
            fun obs ->
                try
                    let v = feat obs |> float
                    if System.Double.IsNaN v
                    then Invalid(obs,"NaN")
                    elif System.Double.IsInfinity v
                    then Invalid(obs,"Infinity")
                    else Valid v
                with
                | ex -> Invalid(obs,ex.Message)
        Continuous(extractor)

    type Features<'Obs> = Feature<'Obs> list

    let binned bins value =
        let bins = bins |> Seq.sort
        let values = 
            bins 
            |> Seq.length
        let activeCase = 
            fun obs ->
                try
                    let value = value obs
                    let index = 
                        bins
                        |> Seq.tryFindIndex (fun b -> value <= b)
                    match index with
                    | Some(index) -> Valid(index)
                    | None -> Valid(values)
                with
                | ex -> Invalid(obs,ex.Message)

        Categorical(values + 1, activeCase)

    let isNumber (x:float) =
        System.Double.IsInfinity x
        || System.Double.IsNaN x
        |> not
    
    let properNumber (x:float) =
        if isNumber x
        then Some x
        else None 

    let encode<'Obs> (feature:Feature<'Obs>) (obs:'Obs) =
        try
            match feature with
            | Continuous(value) -> 
                match (value obs) with
                | Invalid(_) -> None
                | Valid(x) -> Seq.singleton x |> Some
            | Categorical(cases,index) ->
                match (index obs) with
                | Invalid(_) -> None
                | Valid(ind) ->
                    Seq.init 
                        (cases - 1)
                            (fun i -> 
                                if i = ind then 1. else 0.)
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

[<RequireQualifiedAccess>]
module Features = 

    open Core

    let summary sample feature  = 
        
        let count = sample |> Seq.length

        match feature with
        | Continuous(value) ->

            let valid = 
                sample
                |> Seq.map value
                |> extractValid

            let validCount = valid |> Seq.length
            printfn "Values: %i (Valid: %i, Invalid: %i)" count validCount (count - validCount)

            if validCount > 0
            then
                let minimum = valid |> Seq.min
                let maximum = valid |> Seq.max
                let average = valid |> Seq.average

                printfn "Minimum: %.4f" minimum
                printfn "Maximum: %.4f" maximum
                printfn "Average: %.4f" average

        | Categorical(cases,case) ->

            let valid = 
                sample
                |> Seq.map case
                |> extractValid

            let validCount = valid |> Seq.length
            printfn "Values: %i (Valid: %i, Invalid: %i)" count validCount (count - validCount)

            if validCount > 0
            then
                let values =
                    valid
                    |> Seq.countBy id
                    |> dict

                for c in 0 .. cases - 1 do
                    match (values.TryGetValue c) with
                    | true, count -> printfn "Case %i: %i" (c+1) count
                    | false, _    -> printfn "Case %i: %i" (c+1) 0

    let diagnosis sample feature = 

        let count = sample |> Seq.length
        let errors xs f =
            xs
            |> Seq.map f
            |> Seq.filter (function 
                | Invalid(_) -> true 
                | Valid(_) -> false)
            |> Seq.map (function
                | Invalid(obs,msg) -> obs, msg
                | Valid(_) -> failwith "Impossible")

        match feature with
        | Continuous(value) -> errors sample value
        | Categorical(cases,case) -> errors sample case
              
module Measures =

    let included (xs:('Lbl*Option<'Lbl>)seq) =
        xs 
        |> Seq.filter (fun (act,pred) -> 
            Option.isSome pred)
        |> Seq.map (fun (act,pred) ->
            act, pred.Value)

    // proportion of cases where predictor agrees 
    // with the true value.
    let accuracy (xs:('Lbl*Option<'Lbl>)seq) =
        xs
        |> included
        |> Seq.averageBy (fun (act,pred) ->
            if act = pred then 1. else 0.)
