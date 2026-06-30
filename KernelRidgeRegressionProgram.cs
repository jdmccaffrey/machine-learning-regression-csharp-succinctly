using System;
using System.IO;
using System.Collections.Generic;

namespace KernelRidgeRegression
{
  internal class KernelRidgeRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin Kernel Ridge Regression using C# ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data ");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      int[] colsX = new int[] { 0, 1, 2, 3, 4 };
      double[][] trainX = MatLoad(trainFile, colsX, ',', "#");
      double[] trainY =  MatToVec(MatLoad(trainFile, [5], ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, colsX, ',', "#");
      double[] testY = MatToVec(MatLoad(testFile, [5], ',', "#"));

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

      // 2. create model and train using Cholesky inv
      Console.WriteLine("\nCreating KRR model ");
      double gamma = 0.30;   // RBF param
      double alpha = 0.01;  // L2 regularization
      Console.WriteLine("Setting RBF gamma = " + gamma.ToString("F2"));
      Console.WriteLine("Setting L2 alpha =  " + alpha.ToString("F5"));
      KernelRidgeRegressor model = new KernelRidgeRegressor(gamma, alpha);
      Console.WriteLine("Done ");

      Console.WriteLine("\nTraining KRR model using Cholesky inverse ");
      model.Train(trainX, trainY);
      Console.WriteLine("Done ");

      // 3. examine model weights
      Console.WriteLine("\nModel weights: ");
      VecShow(model.weights, 4, 9);

      // 4. evaluate model
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

      // 5. use model
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

    public static void VecShow(double[] vec, int dec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString("F" + dec).PadLeft(wid));
      Console.WriteLine("");
    }
  } // Program

  // ==========================================================================

  public class KernelRidgeRegressor
  {
    public double gamma;  // for RBF kernel
    public double alpha;  // L2 regularization or (wt decay if SGD)
    public double[][] trainX;  // need for any prediction
    public double[] trainY;  // for debugging
    public double[] weights;  // one per trainX item
    private Random rnd;

    // ------------------------------------------------------------------------

    public KernelRidgeRegressor(double gamma, double alpha, int seed = 0)
    {
      this.gamma = gamma;
      this.alpha = alpha;
      this.trainX = new double[0][];  // sort-of null
      this.trainY = new double[0];    // keep compiler happy
      this.weights = new double[0];   // allocated in Train()
      this.rnd = new Random(seed);    // used if SGD training
    }

    // ----------------------------------------------------------------

    public double Predict(double[] x)
    {
      int n = this.trainX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double[] currX = this.trainX[i];
        double k = Rbf(x, currX, this.gamma);
        sum += this.weights[i] * k;
      }
      return sum;
    }

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY)
    {
      // train using Cholesky inverse
      // 0. store trainX -- needed by Predict()
      this.trainX = trainX;  // by ref -- could copy
      this.trainY = trainY;

      // 1. compute train-train K matrix
      int n = trainX.Length;
      double[][] K = MatMake(n, n);
      for (int i = 0; i < n; ++i)
      {
        for (int j = i; j < n; ++j)
        {
          double k = Rbf(trainX[i], trainX[j], this.gamma);
          K[i][j] = k;
          K[j][i] = k;
        }
      }

      // 2. add L2 regularization on diagonal (no bias used)
      for (int i = 0; i < n; ++i)
        K[i][i] += this.alpha;

      // 3. compute model weights using K inverse
      double[][] Kinv = Cholesky.MatInverse(K);  // or MatInverse2(K)
      this.weights = MatVecProduct(Kinv, trainY);
    }

    // ------------------------------------------------------------------------

    private static double[][] MatMake(int nRows, int nCols)
    {
      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
        result[i] = new double[nCols];
      return result;
    }

    // ------------------------------------------------------------------------

    private static double Rbf(double[] v1, double[] v2, double gamma)
    {
      // the gamma version aot sigma / len_scale version
      int dim = v1.Length;
      double sum = 0.0;
      for (int i = 0; i < dim; ++i)
      {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      }
      return Math.Exp(-1 * gamma * sum);
    }
    
    // ------------------------------------------------------------------------

    private static double[] MatVecProduct(double[][] M, double[] v)
    {
      // helper for Train()
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

    public double Accuracy(double[][] dataX, double[] dataY, double pctClose)
    {
      int numCorrect = 0; int numWrong = 0;
      int n = dataX.Length;

      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double predY = this.Predict(x);
        double actualY = dataY[i];
        if (Math.Abs(predY - actualY)
          < Math.Abs(pctClose * actualY))
          numCorrect += 1;
        else
          numWrong += 1;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    // ------------------------------------------------------------------------

    public double MSE(double[][] dataX, double[] dataY)
    {
      double sum = 0.0;
      int n = dataX.Length;
      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double actualY = dataY[i];
        double predY = this.Predict(x);
        sum += (actualY - predY) * (actualY - predY);
      }
      return sum / n;
    }

  } // class KernelRidgeRegressor

  // ==========================================================================

  public class Cholesky
  {
    // container class for MatInverse() via Cholesky decomposition

    public static double[][] MatInverse(double[][] M)
    {
      // conceptually clear version, but slightly less efficient
      // M = L * Lt
      // inv(M) = inv(L * Lt)
      // inv(M) = inv(Lt) * inv(L) is easy uper-tri , lower-tri
      double[][] L = MatDecompCholesky(M);
      double[][] Lt = MatTranspose(L);
      double[][] invL = MatInvLowerTri(L);
      double[][] invLt = MatInvUpperTri(Lt);
      double[][] result = MatProd(invLt, invL);
      return result;
    }

    // ------------------------------------------------------------------------

    public static double[][] MatInverse2(double[][] M)
    {
      // slightly more efficient but not as conceptually clear version
      // M = L * Lt
      // inv(M) = inv(L * Lt)
      // inv(M) = inv(Lt) * inv(L)
      double[][] L = MatDecompCholesky(M);
      double[][] result = MatInverseFromCholesky(L); // all the work
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatInvLowerTri(double[][] L)
    {
      // inverse of lower triangular non-fancy version
      int n = L.Length;  // must be square matrix
      double[][] result = MatIdentity(n);

      for (int k = 0; k < n; ++k)
      {
        for (int j = 0; j < n; ++j)
        {
          for (int i = 0; i < k; ++i)
          {
            result[k][j] -= result[i][j] * L[k][i];
          }
          result[k][j] /= L[k][k]; // check or condition
        }
      }
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatInvUpperTri(double[][] U)
    {
      int n = U.Length;  // must be square matrix
      double[][] result = MatIdentity(n);

      for (int k = 0; k < n; ++k)
      {
        for (int j = 0; j < n; ++j)
        {
          for (int i = 0; i < k; ++i)
          {
            result[j][k] -= result[j][i] * U[i][k];
          }
          result[j][k] /= U[k][k]; // check or condition
        }
      }
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatTranspose(double[][] M)
    {
      int nr = M.Length;
      int nc = M[0].Length;
      double[][] result = MatMake(nc, nr);  // note
      for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[j][i] = M[i][j];
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatDecompCholesky(double[][] M)
    {
      // Cholesky decomposition (Banachiewicz algorithm)
      // M is square, symmetric, positive definite
      // assume M is conditioned on diagonal
      int n = M.Length;
      double[][] result = MatMake(n, n);  // all 0.0
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j <= i; ++j)
        {
          double sum = 0.0;
          for (int k = 0; k < j; ++k)
            sum += result[i][k] * result[j][k];
          if (i == j)
          {
            double tmp = M[i][i] - sum;
            if (tmp < 0.0)
              throw new
                Exception("MatDecompCholesky fatal");
            result[i][j] = Math.Sqrt(tmp);
          }
          else
          {
            if (Math.Abs(result[j][j]) < 1.0e-12)
              throw new
                Exception("MatDecompCholesky div by near zero ");
            result[i][j] =
              (1.0 / result[j][j] * (M[i][j] - sum));
          }
        } // j
      } // i
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatMake(int nRows, int nCols)
    {
      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
        result[i] = new double[nCols];
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatIdentity(int n)
    {
      double[][] result = MatMake(n, n);
      for (int i = 0; i < n; ++i)
        result[i][i] = 1.0;
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatProd(double[][] A, double[][] B)
    {
      int aRows = A.Length;
      int aCols = A[0].Length;
      int bRows = B.Length;
      int bCols = B[0].Length;
      if (aCols != bRows)
        throw new Exception("Non-conformable matrices");

      double[][] result = MatMake(aRows, bCols);
      for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k)
            result[i][j] += A[i][k] * B[k][j];

      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatInverseFromCholesky(double[][] L)
    {
      // L is a lower triangular result of Cholesky decomp
      // direct version
      int n = L.Length;
      double[][] result = MatIdentity(n);

      for (int k = 0; k < n; ++k)
      {
        for (int j = 0; j < n; j++)
        {
          for (int i = 0; i < k; i++)
          {
            result[k][j] -= result[i][j] * L[k][i];
          }
          result[k][j] /= L[k][k]; // should check or condition!
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
          result[k][j] /= L[k][k]; // should check or condition!
        }
      }
      return result;
    }

    // ------------------------------------------------------------------------

  } // class Cholesky

  // ==========================================================================

} // ns
