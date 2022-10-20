using AutoML;
using AutoML.Helpers;
using AutoML.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;



var datasetPath = PathHelper.GetAbsolutePath(@"../../../Dataset/SpamDataset");

var mlContext = new MLContext();
var spamData = mlContext.Data.LoadFromTextFile<SpamInput>(datasetPath, hasHeader: false, separatorChar: ';');

var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
    .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options {
        WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
        CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 3, UseAllLengths = false },
    }, "Message"))
    .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
    .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"))
    .AppendCacheCheckpoint(mlContext);

var preprocessedData = dataProcessPipeline.Fit(spamData).Transform(spamData);

var experimentSettings = new MulticlassExperimentSettings {
    MaxExperimentTimeInSeconds = 60,
    CancellationToken = default,
    CacheDirectoryName = null,
    CacheBeforeTrainer = CacheBeforeTrainer.Auto,
    OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy
};

var progressHandler = new MulticlassExperimentProgressHandler();
var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);
var experimentResult = experiment.Execute(preprocessedData, "Label", progressHandler: progressHandler);
var aaa = 5;