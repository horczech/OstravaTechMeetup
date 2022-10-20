using Microsoft.ML;
using MnistClassification;
using MnistClassification.DataStructures;
using static MnistClassifiation.Runner.Utilities.SampleData;


#region Paths

var modelPath = PathHelper.GetAbsolutePath(@"C:\Users\horakm\Desktop\Prezentace\Demo\MulticlassClassification_MNIST\mnist\MLModels\Model.zip");

#endregion

var mlContext = new MLContext();

var model = mlContext.Model.Load(modelPath, out _);
var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);

var testSamples = GetData();
foreach (var sample in testSamples) {
    var prediction = predictionEngine.Predict(sample.Pixels);

    #region Print Result

    Console.WriteLine("\n\n\t********** TEST PREDICTION RESULT **********");

    Console.WriteLine($"\t0: {prediction.Score[0]:F4}");
    Console.WriteLine($"\t1: {prediction.Score[1]:F4}");
    Console.WriteLine($"\t2: {prediction.Score[2]:F4}");
    Console.WriteLine($"\t3: {prediction.Score[3]:F4}");
    Console.WriteLine($"\t4: {prediction.Score[4]:F4}");
    Console.WriteLine($"\t5: {prediction.Score[5]:F4}");
    Console.WriteLine($"\t6: {prediction.Score[6]:F4}");
    Console.WriteLine($"\t7: {prediction.Score[7]:F4}");
    Console.WriteLine($"\t8: {prediction.Score[8]:F4}");
    Console.WriteLine($"\t9: {prediction.Score[9]:F4}");
    Console.WriteLine("\n\n\t********************************************");
    ShowImage(sample.PathToImage);

    #endregion
}



