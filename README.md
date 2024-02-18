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
Initialization.
```
da = claude.CLAUDE()
```
Estimation of a prediction model from feature data 'X' and target variable data 'Y'.
```
da.fit(X, Y)
```
Calculate prediction for a given feature 'x'. 'x' is allowed to be a scalar, a list of values, or a numpy array.
```
da.predict(x)
```
Calculate linear regression parameters 'W'.
```
da.calculateLinearRegressionParameter()
```
Loading model parameters from an xml file ('parameter_model.xml' by default).
```
da.loadModelParameterFromFile()
```
Loading covariance matrix data from an xml file ('covariance_matrix.xml' by default).
```
da.loadCovarianceMatrixFromFile()
```
Saving model parameters to an xml file.
```
da.saveModelParameterToFile()
```
Saving covariance matrix data to an xml file.
```
da.saveCovarianceMatrixToFile()
```
Updating a covariance matrix from a new data '(x,y)'. 'x' and 'y' must be arrays.
```
da.updateCovarianceMatrix(x, y)
```
Search a precision matrix giving a maximum posterior probability for a given covariance matrix.
```
da.findMaximumPosteriorProbability()
```
## Reference
[Y. Harashima, *et. al.*, Phys. Rev. Materials **5**, 013806 (2021).](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.5.013806)
