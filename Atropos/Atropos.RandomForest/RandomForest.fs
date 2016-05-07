namespace Atropos.RandomForest

[<RequireQualifiedAccess>]
module RandomForest = 

    open Atropos.Utilities
    open Atropos.Core

    type ForestConfig = {
        Trees: int 
        ProportionHeldOut: float 
        FeaturesUsed: int -> int 
        } 

    let defaultHoldout = float >> sqrt >> ceil >> int

    let DefaultRFConfig =  {
        Trees = 50 
        ProportionHeldOut = 0.25 
        FeaturesUsed = defaultHoldout
        } 

    let regression : Learner<'Obs,float> =
        fun model ->
            fun sample ->

                let featurize =
                    model
                    |> learnFeatures sample
                    |> featurizer

                let trainingData =
                    sample
                    |> Seq.map (fun (obs,lbl) ->
                        let fs = featurize obs
                        Array.append fs [| lbl |])
                    // discard any row with missing data
                    |> Seq.filter (numbers)
                    |> array2D

                let sampleSize = trainingData |> Array2D.length1
                // TODO verify. Is this correct, now
                // that discrete features get 'exploded'?
                let featureCount = model |> Seq.length

                let config = DefaultRFConfig
                let featuresUsed = config.FeaturesUsed featureCount

                let _info, forest, forestReport =
                    alglib.dfbuildrandomdecisionforestx1(
                        trainingData, 
                        sampleSize, 
                        featureCount, 
                        1,
                        config.Trees, 
                        featuresUsed, 
                        config.ProportionHeldOut)
                
                let predictor (obs:'Obs) = 
                    let fs = featurize obs
                    let mutable result = Array.empty<float>
                    alglib.dfprocess(forest, fs, &result)
                    result.[0]

                predictor

    let classifier : Learner<'Obs,'Lbl> =
        fun model ->
            fun sample ->

                let featurize =
                    model
                    |> learnFeatures sample
                    |> featurizer

                // identify unique labels
                // should probably also pass in
                // the function to 'create' labels,
                // and define convention for NA
                // (probably Option?)
                let labels =
                    sample
                    |> Seq.map snd
                    |> Seq.distinct 
                    |> Seq.toArray

                // assign arbitrary index to labels
                let labelize (lbl:'Lbl) =
                    labels
                    |> Array.findIndex ((=) lbl)
                    |> float 

                let delabelize (x:float) : 'Lbl =
                    x
                    |> int
                    |> fun index ->
                        if index < 0 || index >= labels.Length
                        then failwith "predicted index should always match to a label"
                        else labels.[index]

                // TODO: include that in some form of a report?
                let trainingData =
                    sample
                    |> Seq.map (fun (obs,lbl) ->
                        let fs = featurize obs
                        let l = labelize lbl
                        Array.append fs [| l |])
                    // discard any row with missing data
                    |> Seq.filter (numbers)
                    |> array2D

                let sampleSize = trainingData |> Array2D.length1
                let featureCount = model |> Seq.length

                let config = DefaultRFConfig
                let featuresUsed = config.FeaturesUsed featureCount

                let _info, forest, forestReport =
                    alglib.dfbuildrandomdecisionforestx1(
                        trainingData, 
                        sampleSize, 
                        featureCount, 
                        labels.Length, // number of classes.
                        config.Trees, 
                        featuresUsed, 
                        config.ProportionHeldOut)
                
                let predictor (obs:'Obs) = 
                    let fs = featurize obs
                    let mutable result = Array.empty<float>
                    alglib.dfprocess(forest, fs, &result)
                    result.[0] |> delabelize

                predictor
