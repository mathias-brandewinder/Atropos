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

    type Predicate<'Obs> = 'Obs -> bool
    // define names for each case, and a predicate
    // that identifies the case.
    // this is expected to be mutually exclusive,
    // and collectively exhaustive.
    type FeatureDefinition<'Obs> = (string * Predicate<'Obs>) list

    // TODO make it consistent so that unexpected labels
    // do not throw. Use TryFind?
    let discreteFrom (f:FeatureDefinition<'Obs>) (obs:'Obs) =
        // extract the case names
        f |> Seq.map fst |> Seq.toArray,
        // ... and the case for the observation.
        f 
        |> Seq.find (fun (_,pred) -> pred obs)
        |> fst

    // transform a categorical feature
    // into columns marked 0 or 1.
    // we create one column less than
    // we have cases, because the last
    // would be redundant.
    // missing/unexpected values result
    // in a NaN-filled array.
    // TODO confirm that 1-less column is OK:
    // definitely right for regression, but
    // what about trees for instance?
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

    let fromDefinition (definition:FeatureDefinition<'Obs>) : FeatureLearner<'Obs,'Lbl> =
        let from = discreteFrom definition
        fun sample ->
            fun pass -> 
                from pass |> Discrete

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
    // transform it into a float[] (a vector).
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

    // create a function that, given a sample
    // and a model, will take an observation
    // and return a vector of features.
    let featuresExtractor model sample =
        model
        |> learnFeatures sample
        |> featurizer
