using Microsoft.ML.Data;

namespace AutoML.Models {
    public class SpamPrediction {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
