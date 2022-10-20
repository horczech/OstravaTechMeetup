using System;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using MnistClassification.DataStructures;

namespace MnistClassification
{
    public class Program
    {
        #region Paths
        private static readonly string DatasetPath = PathHelper.GetAbsolutePath(@"../../../Data/mnist-dataset.csv");
        private static readonly string ModelRelativePath = "../../../MLModels/Model.zip";
        private static readonly string ModelPath = PathHelper.GetAbsolutePath(ModelRelativePath);
        #endregion

        public static void Main()
        {
            var mlContext = new MLContext();
            mlContext.Log += Logger();

            #region STEP 1: Load Data

            var dataset = mlContext.Data.LoadFromTextFile<InputData>(DatasetPath, separatorChar: ',', hasHeader: false);
            var datasetSplit = mlContext.Data.TrainTestSplit(data: dataset, testFraction: 0.2);

            #endregion

            #region STEP 2: Preprocess Data

            var dataProcessPipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Number", outputColumnName: "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.Concatenate(inputColumnNames: "PixelValues", outputColumnName: "Features"))
                .AppendCacheCheckpoint(mlContext);

            #endregion

            #region STEP 3: Create training pipeline

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:"Number", inputColumnName:"Label"));

            #endregion

            #region STEP 4: Train & Save 

            var trainedModel = trainingPipeline.Fit(datasetSplit.TrainSet);
            mlContext.Model.Save(trainedModel, datasetSplit.TrainSet.Schema, ModelPath);

            #endregion

            #region STEP 5: Evaluate

            var predictions = trainedModel.Transform(datasetSplit.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(data:predictions, labelColumnName:"Number", scoreColumnName:"Score");

            Console.WriteLine("\n\n");
            Console.WriteLine($"Evaluation metrics for trained model:\n");
            Console.WriteLine($"AccuracyMacro = {metrics.MacroAccuracy:F4}");
            Console.WriteLine($"AccuracyMicro = {metrics.MicroAccuracy:F4}");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            #endregion
        }

        #region Loggger

        private static EventHandler<LoggingEventArgs> Logger() {
            return (_, args) => { Console.WriteLine(args.Message); };
        }

        #endregion
    }
}
