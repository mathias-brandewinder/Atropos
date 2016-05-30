namespace Atropos

module Core =
    (*
    'Obs = generic Observation type
    'Lbl = generic Label type
    *)

    type Predictor<'Obs,'Lbl> = 'Obs -> 'Lbl

    type Example<'Obs,'Lbl> = 'Obs * 'Lbl

    // Define two types of variables:
    // Discrete: can take one of a few possible values,
    // Continuous: can take any float value.
    type Variable =
        | Discrete of string[] * string
        | Continuous of float
    
    // transform a categorical feature
    // into columns marked 0 or 1.
    // we create one column less than
    // we have cases, because the last
    // would be redundant.
    // missing/unexpected values result
    // in a NaN-filled array.
    // TODO confirm that 1-less column is OK.
    let explode (values:string[]) (value:string) =
        if (values |> Array.contains value)
        then
            Array.init (values.Length - 1) 
                (fun i -> if values.[i] = value then 1. else 0.)
        else
            Array.init (values.Length - 1) (fun _ -> nan)
  
    // A Feature extracts a measure from
    // a single Observation
    type Feature<'Obs> = 'Obs -> Variable

    // TODO: consider a broader notion of 'context',
    // i.e. information available besides sample.
    // A FeatureLearner calibrates a Feature
    // on a Sample.
    type FeatureLearner<'Obs,'Lbl> = 
        Example<'Obs,'Lbl> seq -> 
            Feature<'Obs>

    // A Learner uses a collection of Features
    // and a Sample to learn a Predictor.
    type Learner<'Obs,'Lbl> =
        FeatureLearner<'Obs,'Lbl> seq ->
            Example<'Obs,'Lbl> seq ->
                Predictor<'Obs,'Lbl>

    // Learn/calibrate features on the sample data.
    let learnFeatures =
        fun (sample:Example<'Obs,'Lbl> seq) ->
            fun (learners:FeatureLearner<'Obs,'Lbl> seq) ->
                learners
                |> Seq.map (fun learner -> learner sample)

    // Apply features to an observation, and 
    // transform it into a float[] (a vector)
    let featurizer = 
        fun (features:Feature<'Obs> seq) ->
            fun (obs:'Obs) ->
                features
                |> Seq.map (fun f -> f obs)
                |> Seq.map (fun v ->
                    match v with
                    | Continuous(x) -> [| x |]
                    | Discrete(xs,x) -> explode xs x)
                |> Seq.toArray
                |> Array.collect id
