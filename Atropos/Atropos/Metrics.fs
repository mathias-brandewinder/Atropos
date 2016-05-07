namespace Atropos

module Metrics =

    // Root Mean Square Error
    let RMSE xs =
        xs 
        |> Seq.averageBy (fun (fst,snd) -> 
            pown (fst - snd) 2) 
        |> sqrt