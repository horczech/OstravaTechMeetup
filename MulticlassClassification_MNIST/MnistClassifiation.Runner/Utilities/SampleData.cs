using MnistClassification;
using MnistClassification.DataStructures;
using OpenCvSharp;

namespace MnistClassifiation.Runner.Utilities
{
    public static class SampleData
    {
        public static List<TestData> GetData() {
            var basePath = PathHelper.GetAbsolutePath(@"..\..\..\Images");

            return new List<TestData>() {
                new TestData {
                    Pixels = new InputData(){PixelValues = new float[]{0, 0, 0, 0, 14, 13, 1, 0, 0, 0, 0, 5, 16, 16, 2, 0, 0, 0, 0, 14, 16, 12, 0, 0, 0, 1, 10, 16, 16, 12, 0, 0, 0, 3, 12, 14, 16, 9, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 0, 0, 0, 4, 16, 14, 0, 0, 0, 0, 0, 1, 13, 16, 1, 0}},
                    PathToImage = Path.Combine(basePath, "num1.png"),
                },
                new TestData {
                    Pixels = new InputData(){PixelValues = new float[]{0, 0, 1, 8, 15, 10, 0, 0, 0, 3, 13, 15, 14, 14, 0, 0, 0, 5, 10, 0, 10, 12, 0, 0, 0, 0, 3, 5, 15, 10, 2, 0, 0, 0, 16, 16, 16, 16, 12, 0, 0, 1, 8, 12, 14, 8, 3, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 0}},
                    PathToImage = Path.Combine(basePath, "num7.png"),
                },
                new TestData {
                    Pixels = new InputData(){PixelValues = new float[]{0, 0, 6, 14, 4, 0, 0, 0, 0, 0, 11, 16, 10, 0, 0, 0, 0, 0, 8, 14, 16, 2, 0, 0, 0, 0, 1, 12, 12, 11, 0, 0, 0, 0, 0, 0, 0, 11, 3, 0, 0, 0, 0, 0, 0, 5, 11, 0, 0, 0, 1, 4, 4, 7, 16, 2, 0, 0, 7, 16, 16, 13, 11, 1}},
                    PathToImage = Path.Combine(basePath, "num9.png"),
                }
            };
        }

        public static void ShowImage(string imagePath) {
            var image = Cv2.ImRead(imagePath).Resize(new Size(250,250));
            Cv2.ImShow("Number", image);
            Cv2.WaitKey();
        }
    }
}
