namespace Atropos

module Utilities =

    open System

    let missing = Double.IsNaN
    let infinity = Double.IsInfinity

    let number x =
        (missing x || infinity x) |> not

    // define a default value to replace
    // missing / NA values.    
    let naReplace def x = 
        if number x then x else def

    let numbers xs = xs |> Seq.filter number

    let average xs = xs |> numbers |> Seq.average

    let avgReplace xs =
        (average xs) |> naReplace
