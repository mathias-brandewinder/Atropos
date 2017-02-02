namespace Atropos

module Metrics =

    // Root Mean Square Error
    let RMSE xs =
        xs 
        |> Seq.averageBy (fun (fst,snd) -> 
            pown (fst - snd) 2) 
        |> sqrt

    let accuracy xs =
        xs
        |> Seq.averageBy (fun (val1,val2) ->
            if val1 = val2 
            then 1.0
            else 0.0)
