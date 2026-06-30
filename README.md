# machine-learning-regression-csharp-succinctly
Six machine learning regression techniques implemented from scratch using C#

1.) linear regression, 2.) nearest neighbors regression, 3.) quadratic regression, 4.) kernel ridge regression, 5.) support vector regression, 6.) neural network regression

Example output from one of the linear regression systems:

```
Begin linear regression with SGD training

Loading synthetic train (200) and test (40) data
Done

First three train X:
 -0.6046  0.7260  0.9668 -0.6723  0.1947
  0.9341  0.0945  0.9454  0.4296  0.3955
 -0.9820 -0.2269 -0.9117  0.9133 -0.1277

First three train y:
  0.7180
  0.2507
  0.5698

Creating linear regression model
Done

Training model using SGD
Setting lrnRate = 0.0010
Setting maxEpochs = 1000
epoch =     0  MSE =   0.1898
epoch =   200  MSE =   0.0013
epoch =   400  MSE =   0.0013
epoch =   600  MSE =   0.0013
epoch =   800  MSE =   0.0013
Done

Model weights/coefficients:
-0.2500  -0.0220  0.0272  -0.1434  0.0510
Model bias/intercept: 0.4937

Evaluating model

Accuracy train (within 0.05) = 0.5600
Accuracy test (within 0.05) = 0.5250

MSE train = 0.0013
MSE test = 0.0011

Predicting for x =
  -0.6046   0.7260   0.9668  -0.6723   0.1947
Predicted y = 0.7616

End demo
```

All demo programs use training data **synthetic_train_200.txt** and test data **synthetic_test_40.txt**

**LinearRegressionProgram.cs** - linear regression trained using stochastic gradient descent (SGD)

**LinearRegressionLeftPinvProgram.cs** - linear regression trained using closed-form left pseudo-inverse via Cholesky decomposition

**LinearRegressionPinvQRProgram.cs** - linear regression trained using closed-form Moore-Penrose pseudo-inverse via QR-Householder decomposition
