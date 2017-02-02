// want: recognize discrete cases
// ex: class 1 vs class 2 or 3

type Obs = { Class:int }

type Predicate<'a> = 'a -> bool
type FeatureDefinition<'a> = (string * Predicate<'a>) list

let definition : FeatureDefinition<Obs> =
    [
        "First", fun (obs:Obs) -> obs.Class = 1
        "Second or Third", fun (obs:Obs) -> obs.Class = 2 || obs.Class = 3
    ]

let discreteFrom (f:FeatureDefinition<'a>) (a:'a) =
    f |> Seq.map fst |> Seq.toArray,
    f 
    |> Seq.find (fun (_,pred) -> pred a)
    |> fst

let sample = [
    { Class = 1 }
    { Class = 2 }
    { Class = 3 }
    ]

//let explode (fs:FeatureDefinition<'a>) (obs:'a) =
//    fs 
//    |> List.map (fun (_,pred) -> if pred obs then 1. else 0.)

//sample |> List.map (explode feature)

let foo = discreteFrom definition

sample |> List.map foo
