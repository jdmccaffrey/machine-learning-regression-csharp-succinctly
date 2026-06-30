using System;
using System.IO;
using System.Collections.Generic;

namespace LinearRegression
{
  internal class LinearRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin linear regression with SGD training");
      
      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      double[][] trainX = MatLoad(trainFile, new int[] { 0, 1, 2, 3, 4 },
        ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile, new int[] { 5 },
        ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, new int[] { 0, 1, 2, 3, 4 },
        ',', "#");
      double[] testY = MatToVec(MatLoad(testFile,  new int[] { 5 },
        ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").
          PadLeft(8));

      // 2. create model
      Console.WriteLine("\nCreating linear regression model ");
      LinearRegressor model = new LinearRegressor(seed: 0);
      Console.WriteLine("Done");

      // 3. train model using SGD
      Console.WriteLine("\nTraining model using SGD ");
      double lrnRate = 0.001;
      int maxEpochs = 1000;
      Console.WriteLine("Setting lrnRate = " + lrnRate.ToString("F4"));
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);

      model.Train(trainX, trainY, lrnRate, maxEpochs);
      Console.WriteLine("Done ");

      // 4. examine model parameters
      Console.WriteLine("\nModel weights/coefficients: ");
      for (int i = 0; i < model.weights.Length; ++i)
        Console.Write(model.weights[i].ToString("F4") + "  ");
      Console.WriteLine("\nModel bias/intercept: " +
        model.bias.ToString("F4"));

      // 5. evaluate model
      Console.WriteLine("\nEvaluating model ");

      double accTrain = model.Accuracy(trainX, trainY, 0.05);
      Console.WriteLine("\nAccuracy train (within 0.05) = " +
        accTrain.ToString("F4"));
      double accTest = model.Accuracy(testX, testY, 0.05);
      Console.WriteLine("Accuracy test (within 0.05) = " +
        accTest.ToString("F4"));

      double mseTrain = model.MSE(trainX, trainY);
      Console.WriteLine("\nMSE train = " + mseTrain.ToString("F4"));
      double mseTest = model.MSE(testX, testY);
      Console.WriteLine("MSE test = " + mseTest.ToString("F4"));

      // 6. use model
      double[] x = trainX[0];
      Console.WriteLine("\nPredicting for x = ");
      VecShow(x, 4, 9);
      double predY = model.Predict(x);
      Console.WriteLine("Predicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();

    } // Main()

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

  public class LinearRegressor
  {
    public double[] weights;
    public double bias;
    private Random rnd; // for SGD training

    public LinearRegressor(int seed = 0)
    {
      this.weights = new double[0]; // quasi-null
      this.bias = 0.0;
      this.rnd = new Random(seed);
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      double result = 0.0;
      for (int j = 0; j < x.Length; ++j)
        result += x[j] * this.weights[j];
      result += this.bias;
      return result;
    }

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY,
      double lrnRate, int maxEpochs)
    {
      // basic SGD. no regularization. no early exit.
      int n = trainX.Length;
      int dim = trainX[0].Length;

      this.weights = new double[dim];
      
      int[] indices = new int[n];
      for (int i = 0; i < n; ++i)
        indices[i] = i;

      for (int epoch = 0; epoch < maxEpochs; ++epoch)
      {
        Shuffle(indices, this.rnd);
        for (int i = 0; i < n; ++i)
        {
          int idx = indices[i];
          double[] x = trainX[idx];
          double actualY = trainY[idx];
          double predY = this.Predict(x);

          double err = predY - actualY;

          for (int j = 0; j < dim; ++j)
            this.weights[j] -= lrnRate * err * x[j];
          this.bias -= lrnRate * err;
        }

        // show progress
        if (epoch % (int)(maxEpochs / 5) == 0) // 5 times
        {
          double mse = this.MSE(trainX, trainY);
          string s1 = "epoch = " + epoch.ToString().PadLeft(5);
          string s2 = "  MSE = " + mse.ToString("F4").PadLeft(8);
          Console.WriteLine(s1 + s2);
        }
      }
    } // Train()

    // ------------------------------------------------------------------------

    private static void Shuffle(int[] indices, Random rnd)
    {
      // helper for Train()
      int n = indices.Length;
      for (int i = 0; i < n; ++i)
      {
        int ri = rnd.Next(i, n);
        int tmp = indices[i];
        indices[i] = indices[ri];
        indices[ri] = tmp;
      }
    }

    // ------------------------------------------------------------------------

    public double MSE(double[][] dataX, double[] dataY)
    {
      int n = dataX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double actualY = dataY[i];
        double predY = this.Predict(dataX[i]);
        sum += (actualY - predY) * (actualY - predY);
      }
      return sum / n;
    }

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX, double[] dataY,
      double pctClose)
    {
      int numCorrect = 0; int numWrong = 0;
      for (int i = 0; i < dataX.Length; ++i)
      {
        double actualY = dataY[i];
        double predY = this.Predict(dataX[i]);
        if (Math.Abs(predY - actualY) < Math.Abs(pctClose * actualY))
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numWrong + numCorrect);
    }

    // ------------------------------------------------------------------------

  } // class LinearRegressor

} // ns
