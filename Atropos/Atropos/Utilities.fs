namespace Atropos

module Utilities =

    open System

    // For float values, we will assume that
    // NaN or Infinity are not tolerable values.
    let missing = Double.IsNaN
    let infinity = Double.IsInfinity

    let number x =
        (missing x || infinity x) |> not

    // define a default value to replace
    // missing / NA values.    
    let naReplace def x = 
        if number x then x else def

    // TODO improve naming; goal is to detect
    // if a vector contains only valid numbers.
    let numbers xs = xs |> Seq.forall number

    let average xs = 
        xs 
        |> Seq.filter number 
        |> Seq.average

    let avgReplace (xs:float seq) =
        naReplace (average xs)

    let modeReplace (xs: float seq) = 
        let mode xs = 
            xs 
            |> Seq.filter number
            |> Seq.groupBy id
            |> Seq.maxBy fst
            |> fst
        naReplace (mode xs)
