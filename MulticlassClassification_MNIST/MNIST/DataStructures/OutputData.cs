using Microsoft.ML.Data;

namespace MnistClassification.DataStructures
{
    public class OutputData
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
