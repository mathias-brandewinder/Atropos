namespace Atropos.RandomForest

[<RequireQualifiedAccess>]
module RandomForest = 

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

    let learn : Learner<'Obs,float> =
        fun model ->
            fun sample ->

                let featurize =
                    model
                    |> learnFeatures sample
                    |> featurizer

                let trainInputOutput =
                    sample
                    |> Seq.map (fun (obs,lbl) ->
                        let fs = featurize obs
                        Array.append fs [| lbl |])
                    |> array2D

                let sampleSize = sample |> Seq.length
                let featureCount = model |> Seq.length

                let config = DefaultRFConfig
                let featuresUsed = config.FeaturesUsed featureCount

                let _info, forest, forestReport =
                    alglib.dfbuildrandomdecisionforestx1(
                        trainInputOutput, 
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
