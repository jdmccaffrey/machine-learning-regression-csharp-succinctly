using System;
using System.IO;
using System.Collections.Generic;

namespace QuadraticRegressionSGD
{
  internal class QuadraticRegressionSGDProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin C# quadratic regression ");

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

      // 2. create and train model
      Console.WriteLine("\nCreating quadratic regression model ");
      QuadraticRegressor model = new QuadraticRegressor();
      Console.WriteLine("Done ");

      Console.WriteLine("\nTraining with stochastic gradient descent ");
      double lrnRate = 0.005;
      int maxEpochs = 200;
      Console.WriteLine("\nSetting lrnRate = " + lrnRate.ToString("F4"));
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);
      model.Train(trainX, trainY, lrnRate, maxEpochs);
      Console.WriteLine("Done ");

      // 3. show model weights
      Console.WriteLine("\nModel base weights: ");
      int dim = trainX[0].Length;
      for (int i = 0; i < dim; ++i)
        Console.Write(model.weights[i].ToString("F4").PadLeft(8));
      Console.WriteLine("");

      Console.WriteLine("\nModel quadratic weights: ");
      for (int i = dim; i < dim + dim; ++i)
        Console.Write(model.weights[i].
          ToString("F4").PadLeft(8));
      Console.WriteLine("");

      Console.WriteLine("\nModel interaction weights: ");
      for (int i = dim + dim; i < model.weights.Length; ++i)
      {
        Console.Write(model.weights[i].
          ToString("F4").PadLeft(8));
        if (i > dim + dim && i % dim == 0)
          Console.WriteLine("");
      }
      Console.WriteLine("");

      Console.WriteLine("\nModel bias/intercept: " +
        model.bias.ToString("F4").PadLeft(8));

      // 4. evaluate model
      Console.WriteLine("\nEvaluating model ");
      double accTrain = model.Accuracy(trainX, trainY, 0.05);
      Console.WriteLine("Accuracy train (within 0.05) = " + 
        accTrain.ToString("F4"));
      double accTest = model.Accuracy(testX, testY, 0.05);
      Console.WriteLine("Accuracy test (within 0.05) = " +
        accTest.ToString("F4"));

      double mseTrain = model.MSE(trainX, trainY);
      Console.WriteLine("\nMSE train = " + mseTrain.ToString("F4"));
      double mseTest = model.MSE(testX, testY);
      Console.WriteLine("MSE test = " + mseTest.ToString("F4"));

      // 5. use model
      double[] x = trainX[0];
      Console.WriteLine("\nPredicting for x = ");
      VecShow(x, 4, 9);
      double predY = model.Predict(x);
      Console.WriteLine("\nPredicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main

    // ----------------------------------------------------------------
    // helpers for Main(): MatLoad(), MatToVec(), VecShow()
    // ----------------------------------------------------------------

    static double[][] MatLoad(string fn, int[] usecols,
      char sep, string comment)
    {
      List<double[]> result =
        new List<double[]>();
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

  public class QuadraticRegressor
  {
    public double[] weights;  // regular + quad + interactions
    public double bias;
    private Random rnd;       // for SGD training or noise regularization

    public QuadraticRegressor(int seed = 0)
    {
      this.weights = new double[0];  // keep compiler happy
      this.bias = 0;
      this.rnd = new Random(seed);
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      int dim = x.Length;
      double result = 0.0;

      int p = 0; // points into this.weights
      for (int i = 0; i < dim; ++i)   // regular
        result += x[i] * this.weights[p++];

      for (int i = 0; i < dim; ++i)  // quadratic
        result += x[i] * x[i] * this.weights[p++];

      for (int i = 0; i < dim - 1; ++i)  // interactions
        for (int j = i + 1; j < dim; ++j)
          result += x[i] * x[j] * this.weights[p++];

      result += this.bias;
      return result;
    }
    
    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY,
      double lrnRate, int maxEpochs)
    {
      // train using SGD
      int nRows = trainX.Length;
      int dim = trainX[0].Length;
      int nInteractions = (dim * (dim - 1)) / 2;
      this.weights = new double[dim + dim + nInteractions];

      // technically, not necessary to init weights and bias, but can help
      double low = -0.01; double hi = 0.01;
      for (int i = 0; i < dim; ++i)
        this.weights[i] = (hi - low) * this.rnd.NextDouble() + low;

      this.bias = (hi - low) * this.rnd.NextDouble() + low;

      int[] indices = new int[nRows];
      for (int i = 0; i < nRows; ++i)
        indices[i] = i;

      for (int epoch = 0; epoch < maxEpochs; ++epoch)
      {
        // shuffle order of train data
        int n = indices.Length;
        for (int i = 0; i < n; ++i)
        {
          int ri = this.rnd.Next(i, n);
          int tmp = indices[i];
          indices[i] = indices[ri];
          indices[ri] = tmp;
        }

        for (int i = 0; i < nRows; ++i)
        {
          int idx = indices[i];
          double[] x = trainX[idx];
          double predY = this.Predict(x);
          double actualY = trainY[idx];

          int p = 0; // points into weights

          // update regular weights
          for (int j = 0; j < dim; ++j)
            this.weights[p++] -= lrnRate * (predY - actualY) * x[j];

          // update quadratic weights
          for (int j = 0; j < dim; ++j)
            this.weights[p++] -= lrnRate * (predY - actualY) * x[j] * x[j];

          // update interaction weights
          for (int j = 0; j < dim - 1; ++j)
            for (int k = j + 1; k < dim; ++k)
              this.weights[p++] -= lrnRate * (predY - actualY) * x[j] * x[k];

          // update the bias
          this.bias -= lrnRate * (predY - actualY) * 1.0;
        }
        if (epoch % (int)(maxEpochs / 5) == 0)
        {
          double mse = this.MSE(trainX, trainY);
          string s1 = "epoch = " + epoch.ToString().PadLeft(5);
          string s2 = "  MSE = " + mse.ToString("F4").PadLeft(8);
          Console.WriteLine(s1 + s2);
        }
      }
    } // Train()

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX, double[] dataY, double pctClose)
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
    
  } // class
  
} // ns