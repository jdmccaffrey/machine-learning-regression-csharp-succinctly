using System;
using System.IO;
using System.Collections.Generic;

namespace LinearRegressionLeftPinv
{
  internal class LinearRegressionLeftPinvProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin linear regression with left " +
        "pseudo-inverse training ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train" +
        " (200) and test (40) data");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      int[] colsX = new int[] { 0, 1, 2, 3, 4 };
      int colY = 5;

      double[][] trainX = MatLoad(trainFile, colsX, ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile,
        new int[] { colY }, ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, colsX, ',', "#");
      double[] testY = MatToVec(MatLoad(testFile,
        new int[] { colY }, ',', "#"));
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
      LinearRegressor model = new LinearRegressor();
      Console.WriteLine("Done");

      // 3. train model
      Console.WriteLine("\nTraining model using" +
        " left pseudo-inverse via normal equations ");
      model.Train(trainX, trainY);
      Console.WriteLine("Done ");

      // 4. examine model parameters
      Console.WriteLine("\nWeights/coefficients: ");
      for (int i = 0; i < model.weights.Length; ++i)
        Console.Write(model.weights[i].ToString("F4") + "  ");
      Console.WriteLine("\nBias/constant: " +
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
      Console.WriteLine("Predicted y = " +
        predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main

    // ------------------------------------------------------
    // helpers for Main(): MatLoad, MatToVec, VecShow
    // ------------------------------------------------------

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
    private Random rnd; // not needed for left pinv training

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

    public void Train(double[][] trainX, double[] trainY)
    {
      // train using left pinv via normal equations
      // (using Cholesky inverse)
      // can fail for very large trainX (then use SGD)
      int dim = trainX[0].Length;
      this.weights = new double[dim];

      double[][] X = Cholesky.MatToDesign(trainX);
      double[][] Xpinv = Cholesky.MatPseudoInv(X);
      double[] biasAndWts = Cholesky.MatVecProduct(Xpinv, trainY);
      this.bias = biasAndWts[0];
      for (int i = 1; i < biasAndWts.Length; ++i)
        this.weights[i - 1] = biasAndWts[i];
      return;
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

  } // class LinearRegressor

  // ==========================================================================

  public class Cholesky
  {
    // container class for MatPseudoInv()
    // left pseudo-inv via normal equations, using Cholesky inverse

    public static double[][] MatPseudoInv(double[][] A)
    {
      // left pseudo-inverse via normal equations
      // A is design matrix where nRows must be gte nCols
      // pinv(A) = inv(At * A) * A
      // calls MatInvCholesky, calls MatDecompCholesky
      double[][] At = MatTranspose(A);
      double[][] AtA = MatProduct(At, A); // this could fail
      for (int i = 0; i < AtA.Length; ++i)
        AtA[i][i] += 1.0e-8; /// condition AtA before inv
      double[][] AtAinv = MatInvCholesky(AtA);
      double[][] pinv = MatProduct(AtAinv, At);
      return pinv;
    }

    // ------------------------------------------------------------------------
    // helpers for training that uses MatPseudoInv(): 
    // MatToDesign(), MatVecProduct()
    // ------------------------------------------------------------------------

    public static double[][] MatToDesign(double[][] X)
    {
      // utility function
      // construct design matrix from X (add col of 1.0s)
      int nRows = X.Length; // aka m
      int nCols = X[0].Length; // n
      
      double[][] result = MatMake(nRows, nCols + 1);
      for (int i = 0; i < nRows; ++i)
      {
        result[i][0] = 1.0;
        for (int j = 1; j < nCols + 1; ++j)
          result[i][j] = X[i][j - 1];
      }
      return result;
    }

    // ------------------------------------------------------------------------

    public static double[] MatVecProduct(double[][] M,
      double[] v)
    {
      // M * v. return a regular vector
      int nRows = M.Length;
      int nCols = M[0].Length;
      int n = v.Length;
      if (nCols != n)
        throw new Exception("non-conform in MatVecProd");

      double[] result = new double[nRows];
      for (int i = 0; i < nRows; ++i)
        for (int k = 0; k < nCols; ++k)
          result[i] += M[i][k] * v[k];

      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatInvCholesky(double[][] A)
    {
      // helper for MatPseudoInv()
      // A must be square, symmetric, positive definite
      // calls MatDecompCholesky()
      int m = A.Length;
      int n = A[0].Length;  // m == n
      double[][] L = MatDecompCholesky(A);  // A = L * Lt
      // inv(A) = inv(Lt) * inv(L)

      // optional safety: condition L
      for (int i = 0; i < L.Length; ++i)
        L[i][i] += 1.0e-8;

      // compute inv(Lt) * inv(L) directly without using helpers
      // MatInvUpperTri() and MatInvLowerTri()
      double[][] result = MatMake(n, n); // Identity
      for (int i = 0; i < n; ++i)
        result[i][i] = 1.0;

      for (int k = 0; k < n; ++k)
      {
        for (int j = 0; j < n; j++)
        {
          for (int i = 0; i < k; i++)
          {
            result[k][j] -= result[i][j] * L[k][i];
          }
          result[k][j] /= L[k][k];
        }
      }

      for (int k = n - 1; k >= 0; --k)
      {
        for (int j = 0; j < n; j++)
        {
          for (int i = k + 1; i < n; i++)
          {
            result[k][j] -= result[i][j] * L[i][k];
          }
          result[k][j] /= L[k][k];
        }
      }
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatDecompCholesky(double[][] A)
    {
      // helper for MatInvCholesky()
      // decompose A to L such that L * Lt = A
      // A must be square
      int m = A.Length; int n = A[0].Length;  // m == n
      double[][] L = MatMake(n, n);  // or m

      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j <= i; ++j)
        {
          double sum = 0.0;
          for (int k = 0; k < j; ++k)
            sum += L[i][k] * L[j][k];
          if (i == j)
          {
            double tmp = A[i][i] - sum;
            if (tmp < 0.0)
              throw new
                Exception("decomp Cholesky fatal");
            L[i][j] = Math.Sqrt(tmp);
          }
          else
          {
            if (L[j][j] == 0.0) // or condition
              throw new
                Exception("decomp Cholesky fatal ");
            L[i][j] = (A[i][j] - sum) / L[j][j];
          }
        } // j
      } // i

      return L;
    }

    // ------------------------------------------------------------------------
    // misc. helpers: MatMake(), MatTranspose(), MatProduct()
    // ------------------------------------------------------------------------

    private static double[][] MatMake(int nRows, int nCols)
    {
      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
        result[i] = new double[nCols];
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatTranspose(double[][] M)
    {
      int nRows = M.Length; int nCols = M[0].Length;
      double[][] result = new double[nCols][]; // note
      for (int i = 0; i < nCols; ++i)
        result[i] = new double[nRows];
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[j][i] = M[i][j]; // note
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatProduct(double[][] A, double[][] B)
    {
      int aRows = A.Length; int aCols = A[0].Length;
      int bRows = B.Length; int bCols = B[0].Length;
      if (aCols != bRows)
        throw new Exception("Non-conformable matrices");

      double[][] result = new double[aRows][];
      for (int i = 0; i < aRows; ++i)
        result[i] = new double[bCols];

      for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k)
            result[i][j] += A[i][k] * B[k][j];
      return result;
    }

    // ------------------------------------------------------------------------

  } // class Cholesky
} // ns