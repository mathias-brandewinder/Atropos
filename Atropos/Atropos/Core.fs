namespace Atropos

module Core =

    (*
    'Obs = generic Observation type
    'Lbl = generic Label type
    *)

    type Predictor<'Obs,'Lbl> = 'Obs -> 'Lbl

    type Example<'Obs,'Lbl> = 'Obs * 'Lbl

    // TODO: refine with additional feature information,
    // for instance discrete vs. continuous, or interval
    // (bounded/unbounded).
    // A Feature extracts a measure from
    // a single Observation
    type Feature<'Obs> = 'Obs -> float

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
                |> Seq.toArray