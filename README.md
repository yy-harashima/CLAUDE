# CLAUDE
Covariance Linkage Assimilation for Unobserved Data Exploration
## Installation
Installation by pip from github.
```
pip install git+https://github.com/yy-harashima/CLAUDE.git
```
Update.
```
pip install git+https://github.com/yy-harashima/CLAUDE.git -U
```
## Usage

### Sample code
A sample data. 
Copy these data into a file, e.g., `exampledata.csv`.
```
descriptor,      target1,         target2
0.6942910000,    1.4494230027,             nan
0.2198329578,    2.5198085612,             nan
0.9964402084,    3.9693563738,             nan
0.1134539995,    2.4388244556,             nan
0.4102801276,    1.9849420281,             nan
0.2229659544,    2.4879431009,             nan
0.2327652708,    2.4727410086,             nan
0.9861928108,    3.6604668853,             nan
0.7876733331,    1.6664465466,             nan
0.4402319002,    1.8651530271,             nan
0.4969327180,             nan,    0.1921072673
0.0647297202,             nan,    0.8317566123
0.0988111466,             nan,    0.7394230791
0.1435566698,             nan,    0.9501056826
0.1985281366,             nan,    0.8433509187
```
A sample code for data assimilation.
```
import claude
import numpy as np
import pandas as pd

da = claude.CLAUDE(nDescriptor = 1, \
                   nTarget = 2, \
                   degreePolynomialDescriptor = np.array([[0],[1],[2],[3]]), \
                   degreeActiveDescriptor = [[[0],[0,1]]])

df = pd.read_csv('exampledata.csv', skipinitialspace=True)
xSample = df[['descriptor']].values
ySample = df[['target1','target2']].values

da.fit(xSample,ySample)

x = np.arange(0.0,1.0+1e-6,0.01)
y = da.predict(x)

aq = da.acquisitionFunction(x)
idxMax1 = np.argmax(aq[:,0])
idxMax2 = np.argmax(aq[:,1])

da.saveModelParameterToFile()
da.saveCovarianceMatrixToFile()
```
![Results from the data assimilation](https://github.com/yy-harashima/CLAUDE/blob/main/image/predictioncurve.png)

### Initialization 
```
claude.CLAUDE()
```
or
```
claude.CLAUDE(nDescriptor = 1, \
              nTarget = 2)
```
CLAUDE has two types of initialization procedures:
either initialization parameters are provided directly, 
or they are loaded from XML files.
When variables `nDescriptor` and `nTarget` are not given at the initialization, 
it indicates the direct initialization.
When they are not given, most of internal variables in CLAUDE are not initialized and 
users must call both `loadModelParameterFromFile()` and `loadCovarianceMatrixFromFile()`.  

#### Input parameters  
##### `degreePolynomialDescriptor` (default: None)
list of exponents in a polynomial of descriptors. For example, `[[0],[1],[2]]` indicates the model contains a constant term, a linear term, and a second order term, $y = a_{0} + a_{1}x + a_{2}x^{2}$. Another example is two dimensional case, $x_{1}$ and $x_{2}$. Then, `[[0,0],[1,0],[2,0],[1,1]]` corresponds to $y = a_{00} + a_{10}x_{1} + a_{20}x_{1}^{2} + a_{11}x_{1}x_{2}$.

##### `degreePolynomialTarget` (default: None)
list of exponents in a polynomial of target variables. Now only linear terms, e.g., `[[1,0],[0,1]]` for `nTarget = 2`, are allowed.

##### `degreeActiveDescriptor` (default: None)
Pairs of exponents of descriptors and target variables to use correction terms $R(x)$ of $y_{2}(x) = C y_{1}(x) + R(x)$.
For example, `[[[0],[0,1]], [[1],[0,1]]` indicates $R(x) = r_{0} + r_{1}x$.

##### `degreeInactiveDescriptor` (default: None)
Almost same as `degreeActiveDescriptor`, but this variable specifies inactivation.

##### `minimizationMethod` (default: L-BFGS-B)
The method to find a precision matrix $\Lambda$ by optimizing a posterior distribution.

##### `nSampleMonteCarlo` (default: 500)
The number of Monte Carlo steps for a calculation of acquisition function.

##### `modeAcquisitionFunction` (default: None)
Specifying acquisition functions for each target variable.
When this is not given, the last component of target variables is Expected Improvement (`1`), and the other components are Standard Deviation (`0`).

#### Return  
##### Instance with a type `CLAUDE`

### Fitting data
Estimation of a prediction model from feature data `X` and target variable data `Y`.
```
fit(X, Y)
```
##### `X`
Array of descriptor values.
`X` must be a numpy array, whose shape is `[nSample, nComponent]`.

##### `Y`
Array of target variable values.
The shape is same as `X`.

### Prediction
Calculate prediction for a given feature 'x'. 
```
predict(x)
```
#### Input
##### `x`
This is allowed to be a scalar, a list of values, or a numpy array.

#### Return
##### numpy array
The array of predicted values.

### Linear regression parameters
Calculate a coefficient matrix of linear regression $W$ in $y=Wx$.
```
calculateLinearRegressionParameter()
```

### Load and save model parameters and matrix element data
Loading model parameters from an xml file ('parameter_model.xml' by default).
```
loadModelParameterFromFile()
```
Loading covariance matrix data from an xml file ('covariance_matrix.xml' by default).
```
loadCovarianceMatrixFromFile()
```
Saving model parameters to an xml file.
```
saveModelParameterToFile()
```
Saving covariance matrix data to an xml file.
```
saveCovarianceMatrixToFile()
```

### Covariance matrix update
Updating a covariance matrix from a new data '(x,y)'. 'x' and 'y' must be arrays.
```
updateCovarianceMatrix(x, y)
```

### Maximum a posteriori
Search a precision matrix giving a maximum posterior probability for a given covariance matrix.
```
findMaximumPosteriorProbability()
```

### Acquisition function
Calculate values of acquisition functions.
```
acquisitionFunction(X)
```
#### Input
##### `X`
List of descriptors that values of acquisition functions are calculated.
The array shape is `(nMesh, nComponent)`.

#### Return
##### numpy array
A numpy array containing values of acquisition function.
The array shape is `(nMesh, nTarget)`

## Reference
[Y. Harashima, *et. al.*, Phys. Rev. Materials **5**, 013806 (2021).](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.5.013806)  
[Y. Harashima, *et. al.*, arXiv:2408.08539 (2024).](https://doi.org/10.48550/arXiv.2408.08539)