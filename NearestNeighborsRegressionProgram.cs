using System;
using System.IO;
using System.Collections.Generic;

namespace NearestNeighborsRegression
{
  internal class NearestNeighborsRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin nearest neighbors regression using C# ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data ");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      int[] colsX = new int[] { 0, 1, 2, 3, 4 };
      double[][] trainX = MatLoad(trainFile, colsX, ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile, [5], ',', "#"));
      
      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, colsX, ',', "#");
      double[] testY = MatToVec(MatLoad(testFile, [5], ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

      // 2. create and train/fit model
      Console.WriteLine("\nCreating and training k-NN regression model ");
      int numNeighbors = 4;
      Console.WriteLine("Setting numNeighbors = " + numNeighbors);
      NearestNeighborsRegressor model = 
        new NearestNeighborsRegressor(numNeighbors);
      model.Train(trainX, trainY); // aka Fit()
      Console.WriteLine("Done ");

      // 3. evaluate model
      double accTrain = model.Accuracy(trainX, trainY, 0.05);
      Console.WriteLine("\nAccuracy train (within 0.05): " +
        accTrain.ToString("F4"));
      double accTest = model.Accuracy(testX, testY, 0.05);
      Console.WriteLine("Accuracy test (within 0.05): " +
        accTest.ToString("F4"));

      double mseTrain = model.MSE(trainX, trainY);
      Console.WriteLine("\nMSE train = " +  mseTrain.ToString("F4"));
      double mseTest = model.MSE(testX, testY);
      Console.WriteLine("MSE test = " + mseTest.ToString("F4"));

      // 4. use model
      Console.WriteLine("\nPredicting for trainX[0] ");
      double[] x = trainX[0];
      double y = model.Predict(x);
      Console.WriteLine("Predicted y = " +
        y.ToString("F4").PadLeft(4));

      Console.WriteLine("\nExplaining prediction for x = train[0] ");
      model.Explain(x);

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main

    // ----------------------------------------------------------------
    // helpers for Main(): MatLoad, MatToVec, VecShow
    // ----------------------------------------------------------------

    static double[][] MatLoad(string fn, int[] usecols,
      char sep, string comment)
    {
      List<double[]> result = new List<double[]>();
      string line = "";
      FileStream ifs = new FileStream(fn, FileMode.Open);
      StreamReader sr = new StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
      {
        if (line.StartsWith(comment) == true)
          continue;
        string[] tokens = line.Split(sep);
        List<double> lst = new List<double>();
        for (int j = 0; j < usecols.Length; ++j)
          lst.Add(double.Parse(tokens[usecols[j]]));
        double[] row = lst.ToArray();
        result.Add(row);
      }
      sr.Close(); ifs.Close();
      return result.ToArray();
    }

    static double[] MatToVec(double[][] M)
    {
      int nRows = M.Length;
      int nCols = M[0].Length;
      double[] result = new double[nRows * nCols];
      int k = 0;
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[k++] = M[i][j];
      return result;
    }

    static void VecShow(double[] vec, int dec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString("F" + dec).
          PadLeft(wid));
      Console.WriteLine("");
    }

  } // class Program

  // ==========================================================================

  public class NearestNeighborsRegressor
  {
    public int nNeighbors;
    public double[] weights;
    public double[][] trainX;
    public double[] trainY;

    // ------------------------------------------------------------------------

    public NearestNeighborsRegressor(int nNeighbors)
    {
      this.nNeighbors = nNeighbors;
      this.weights = new double[nNeighbors];
      for (int k = 0; k < nNeighbors; ++k)
        this.weights[k] = 1.0 / nNeighbors;  // wts must sum to 1
      this.trainX = new double[0][]; // quasi-null
      this.trainY = new double[0];   // keep compiler happy
    }

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY)
    {
      this.trainX = trainX; // store by ref
      this.trainY = trainY;
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      // 1. compute distances from x to all trainX
      int N = trainX.Length;
      double[] dists = new double[N];
      for (int i = 0; i < N; ++i)
         dists[i] = Distance(x, this.trainX[i]);
 
      // 2. determine sort order of distances
      int[] sortedIdxs = ArgSort(dists);

      // 3. average of nearest target y values
      double sum = 0.0;
      for (int k = 0; k < this.nNeighbors; ++k)
      {
        int idx = sortedIdxs[k]; // near to far
        sum += this.trainY[idx] * this.weights[k]; // wts must sum to 1
      }
      return sum;
    }

    // ------------------------------------------------------------------------

    public void Explain(double[] x)
    {
      // 0. set up ordering/indices
      int n = this.trainX.Length;
      int[] indices = new int[n];
      for (int i = 0; i < n; ++i)
        indices[i] = i;

      // 1. compute distances from x to all trainX
      double[] distances = new double[n];
      for (int i = 0; i < n; ++i)
        distances[i] = Distance(x, this.trainX[i]);

      // 2. sort distances, and indices of X and Y, by distances
      Array.Sort(distances, indices);

      // 3. compute predicted y
      double predY = 0.0;
      for (int k = 0; k < this.nNeighbors; ++k)
        predY += this.weights[k] * this.trainY[indices[k]];

      // 4. show info 
      for (int k = 0; k < this.nNeighbors; ++k)
      {
        int i = indices[k];
        Console.Write("X = ");
        Console.Write("[" + i.ToString().PadLeft(3) + "] ");
 
        for (int j = 0; j < this.trainX[i].Length; ++j)
          Console.Write(this.trainX[i][j].ToString("F2").PadLeft(6));

        Console.Write(" | y = ");
        Console.Write(this.trainY[i].ToString("F4"));
        Console.Write(" | dist = ");
        Console.Write(distances[k].ToString("F4"));
        Console.WriteLine("");
      }

      Console.WriteLine("\nPredicted y = " +
        predY.ToString("F4"));
    } // Explain

    // ----------------------------------------------------------------

    public double Accuracy(double[][] dataX,
      double[] dataY, double pctClose)
    {
      int nCorrect = 0; int nWrong = 0;
      for (int i = 0; i < dataX.Length; ++i)
      {
        double predY = this.Predict(dataX[i]);
        double actualY = dataY[i];
        if (Math.Abs(predY - actualY) < Math.Abs(pctClose * actualY))
          ++nCorrect;
        else
          ++nWrong;
      }
      return (nCorrect * 1.0) / (nCorrect + nWrong);
    }

    // ----------------------------------------------------------------

    public double MSE(double[][] dataX, double[] dataY)
    {
      double sum = 0.0;
      for (int i = 0; i < dataX.Length; ++i)
      {
        double predY = this.Predict(dataX[i]);
        double actualY = dataY[i];
        sum += (predY - actualY) * (predY - actualY);
      }
      return sum / dataX.Length;
    }

    // ----------------------------------------------------------------

    private double Distance(double[] v1, double[] v2)
    {
      // Euclidean distance
      int n = v1.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      return Math.Sqrt(sum);
    }

    // ------------------------------------------------------------------------

    private static int[] ArgSort(double[] values)
    {
      int n = values.Length;
      double[] copy = new double[n];
      int[] indices = new int[n];
      for (int i = 0; i < n; ++i)
      {
        copy[i] = values[i];
        indices[i] = i;
      }
      Array.Sort(copy, indices); // unique to C#
      return indices;
    }

    // ------------------------------------------------------------------------

  } // class NearestNeighborsRegressor

} // ns
