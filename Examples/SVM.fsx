(*
Classification example, using Accord Support Vector Machine
*)

#load "Dependencies.fsx"
#r "Accord/lib/net46/Accord.dll"
#r "Accord.Math/lib/net46/Accord.Math.dll"
#r "Accord.MachineLearning/lib/net46/Accord.MachineLearning.dll"
#r @"Accord.Statistics/lib/net46/Accord.Statistics.dll"

module SVM =

    open Atropos.Utilities
    open Atropos.Core
    open Accord.MachineLearning.VectorMachines
    open Accord.MachineLearning.VectorMachines.Learning
    open Accord.Statistics.Kernels

    type Config = { 
        Kernel:IKernel
        Tolerance:float
        Complexity:float Option // Some(c) indicates a user-defined value
        }

    let DefaultConfig = { 
        Kernel = Linear()
        Tolerance = 1e-2
        Complexity = None }
    
    let classification : Config -> Learner<'Obs,'Lbl> =

        fun config ->
            fun model ->
                fun sample ->

                    let featurize = featuresExtractor model sample

                    let cases =
                        sample
                        |> Seq.map snd
                        |> Seq.distinct 
                        |> Seq.toArray

                    let normalizer (lbl:'Lbl) = 
                        cases 
                        |> Array.findIndex (fun x -> x = lbl)

                    let denormalizer i = cases.[i] 

                    let features, labels =
                        sample
                        |> Seq.map (fun (obs,lbl) ->
                            let fs = featurize obs
                            fs,lbl |> normalizer)
                        // discard any row with missing data
                        |> Seq.filter (fun (obs,lbl) -> numbers obs)
                        // TODO: can I have one-shot Array extraction?
                        |> Seq.toArray
                        |> Array.unzip

                    let classes = labels |> Array.distinct |> Array.length
                    let kernel = config.Kernel

                    let featuresCount = features.[0].Length

                    let algorithm = 
                        fun (svm: KernelSupportVectorMachine) 
                            (classInputs: float[][]) 
                            (classOutputs: int[]) (i: int) (j: int) -> 
                            
                            let strategy = SequentialMinimalOptimization(svm, classInputs, classOutputs)
                            strategy.Tolerance <- config.Tolerance
                            match config.Complexity with
                            | Some(complexity) ->
                                strategy.Complexity <- complexity
                            | None -> ignore ()
                            
                            strategy :> ISupportVectorMachineLearning

                    let svm = new MulticlassSupportVectorMachine(featuresCount, kernel, classes)
                    let learner = MulticlassSupportVectorLearning(svm, features, labels)
                    let config = SupportVectorMachineLearningConfigurationFunction(algorithm)
                    learner.Algorithm <- config

                    let error = learner.Run()
                            
                    let predictor = 
                        featurize >> svm.Compute >> denormalizer

                    predictor






#load "ClassificationFeatures.fsx"
open ClassificationFeatures

open Atropos.Metrics
open Accord.Statistics.Kernels

let config = { 
    SVM.DefaultConfig with 
        Tolerance = 1e-3
        Kernel = Gaussian ()
        Complexity = Some 5.0 }

let svmClassifier = 
    sample
    |> SVM.classification config model 

// probably an issue here: what is predicted
// when there is an issue with featurization,
// i.e. the vector contains NaNs?
// TODO figure out how to handle.
sample
|> Seq.map (fun (o,l) -> 
    svmClassifier o, l)
|> Seq.map (fun (p,l) -> 
    printfn "Pred: %b, Act: %b" p l
    p,l)
|> accuracy
