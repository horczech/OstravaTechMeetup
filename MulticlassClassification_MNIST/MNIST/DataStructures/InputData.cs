﻿using Microsoft.ML.Data;

namespace MnistClassification.DataStructures
{
    public class InputData
    {
        [ColumnName("PixelValues")]
        [LoadColumn(0, 63)]
        [VectorType(64)]
        public float[] PixelValues;

        [LoadColumn(64)]
        public float Number;
    }
}
