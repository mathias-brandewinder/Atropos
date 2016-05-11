namespace Atropos.RandomForest

[<RequireQualifiedAccess>]
module RandomForest = 

    open Atropos.Utilities
    open Atropos.Core

    type Config = {
        // how many trees to include in the forest
        Trees: int 
        // proportion examples held out in each tree
        ProportionHeldOut: float 
        // number of features included in each tree
        FeaturesUsed: int -> int 
        } 

    // by default we take square root of # of features
    let defaultHoldout = float >> sqrt >> ceil >> int

    let DefaultConfig =  {
        Trees = 50 
        ProportionHeldOut = 0.25 
        FeaturesUsed = defaultHoldout
        } 

    let regression : Config -> Learner<'Obs,float> =
        fun config ->
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

    let classifier : Config -> Learner<'Obs,'Lbl> =

        fun config ->
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
