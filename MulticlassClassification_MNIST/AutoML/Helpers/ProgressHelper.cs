using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace AutoML.Helpers
{
    public class MulticlassExperimentProgressHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
    {
        private int _iterationIndex;

        public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult) {
            Console.WriteLine(
                $"Iteration index: {_iterationIndex}; Trainer: {iterationResult.TrainerName}; Validation metrics: {iterationResult.ValidationMetrics}; RuntimeInSeconds: {iterationResult.RuntimeInSeconds}");
        }
    }
}