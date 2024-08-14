import sys, os
import numpy as np
from xml.dom import minidom
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.linalg import inv, det
import copy
import time

###---------------------------------------------------------------------------
# Flag of debugging for posterior distribution of precision matrixes.
# If True, precision matrix components are saved in the file.
###---------------------------------------------------------------------------
flagDebugLambda = False
filenameDebugLambda = 'debug_lambda.dat'
flagDebugWidthMonteCarlo = False
filenameDebugWidthMonteCarlo = 'debug_width_monte_carlo.dat'
flagDebugPostProb = False
filenameDebugPostProb = 'debug_posteriori_probability.dat'

class CLAUDE:
    ### Default filenames.
    filenameModelParameter0 = 'parameter_model.xml'
    filenameCovarianceMatrix0 = 'covariance_matrix.xml'

    ### Tag names of model parameters and covariance matrix in xml format.
    tagNameModelParameter = 'CLAUDEModelParameter'
    tagNameCovarianceMatrix = 'CLAUDECovarianceMatrix'

    ### Use basin hopping, or not.
    useBasinHopping = False
    iterationBasinHopping = 20

    ### Initial maximal likely precision matrix.
    modeInitialPrecisionMatrixMax0 = 0
    offsetInitialPrecisionMatrixMax0 = 0.1

    ###---------------------------------------------------------------------------
    # Minimization options. They are passed to scipy.optimize.minimization, 
    #       or. scipy.optimize.basinhopping.
    #    method           : 'L-BFGS-B', 'SLSQP', etc.
    #    tol              : float, e.g., 1e-8.
    #    options[maxiter] : integer, e.g., 1000.
    #    options[ftol]    : float, e.g., 1e-10.
    #    options[gtol]    : float, e.g., 1e-8.
    #    options[eps]     : float, e.g., 1e-8.
    #    options[disp]    : True or False.
    ###---------------------------------------------------------------------------
    minimizationMethod0 = 'L-BFGS-B'
    minimizationTolerance0 = 1e-8
    minimizationOptions0 = {'maxiter':1000, \
                            'ftol':1e-10, \
                            'gtol':1e-8, \
                            'eps':1e-8, \
                            'disp':False}

    ### Default values Monte Carlo sampling.
    widthMeanMonteCarlo0 = 100.0
    nSampleMonteCarlo0 = 500

    ###---------------------------------------------------------------------------
    # __init__ : Constructor.
    # 
    # degreePolynomialDescriptor : [[0,0,0], ..., [0,0,2]].
    #       Degrees of polynomials for each descriptor. This is applied for all
    #       target variables. For example, [[a,b,c],[d,e,f]] means 
    #       coef_abc * x0^a x1^b x2^c + coef_def * x0^d x1^e x2^f, where 
    #       x0, x1, x2 are the descriptor.
    #       Default value is [[0,0,0],[1,0,0],[0,1,0],[0,0,1]] in case of three 
    #       descriptors.
    #
    # degreePolynomialTarget : Same expression as degreePolynomialDescriptor, 
    #       but for target variables.
    #       Default value is [[[1,0],[0,1]]].
    #       Note that diagonal components are automatically activated.
    #       In contrast to degreePolynomialTarget, [0,0] is not included.
    #
    # degreeActiveDescriptor : Pairs of degree for descriptors and degree for 
    #       target variables. ex.) [[[0,0,0],[0,1]],[[0,0,1],[0,1]]]. This
    #       means that the degree [0,0,0] of descriptor for the target of [0,1]
    #       and [0,0,1] for the target of [0,1] are activated.
    #
    # degreeActiveTarget : ex.) [[[1,0],[1,0]], [[1,0],[0,1]]].
    #
    # degreeInactiveDescriptor, degreeInactiveTarget : Same as 
    #       degreeActiveDescriptor, degreeActiveTarget.
    #
    # domainDescriptor: Domain of descriptor. ex.) [[0.0,1.0],[0.0,1.0],[0.0,1.0]].
    #
    # filenameModelParameter : The filename containing and saving the model 
    #       parameters. The parameters fixed among the simulation is given.
    #       The parameters updated by data is contained in the file of 
    #       covariance matrix.
    #
    # filenameCovarianceMatrix : The filename for the data of covariance matrixes.
    ###---------------------------------------------------------------------------
    def __init__(self, \
                 nDescriptor = 0, \
                 nTarget = 0, \
                 degreePolynomialDescriptor = None, \
                 degreePolynomialTarget = None, \
                 degreeActiveDescriptor = None, \
                 degreeActiveTarget = None, \
                 degreeInactiveDescriptor = None, \
                 degreeInactiveTarget = None, \
                 domainDescriptor = None, \
                 filenameModelParameter = filenameModelParameter0, \
                 filenameCovarianceMatrix = filenameCovarianceMatrix0, \
                 flagLinearModel = False, \
                 minimizationMethod = minimizationMethod0, \
                 minimizationTolerance = minimizationTolerance0, \
                 minimizationOptions = minimizationOptions0, \
                 modeInitialPrecisionMatrixMax = modeInitialPrecisionMatrixMax0, \
                 offsetInitialPrecisionMatrixMax = offsetInitialPrecisionMatrixMax0, \
                 # constrain = None, \
                 # mode = 2, \
                 # shiftWidth = 0.1, \
                 # standardizeDescriptor = None, \
                 widthMeanMonteCarlo = widthMeanMonteCarlo0, \
                 widthMonteCarlo = None, \
                 nSampleMonteCarlo = nSampleMonteCarlo0, \
                 modeAcquisitionFunction = None, \
                 indexTargetPrimary = -1):

        labelFunction = 'CLAUDE'

        ## Default settings of model parameters.
        self.nDescriptor = nDescriptor
        self.dimensionDescriptor = 0
        self.degreePolynomialDescriptor = None
        self.nTarget = nTarget
        self.dimensionTarget = 0
        self.degreePolynomialTarget = None
        self.activeVariable = None
        self.domainDescriptor = None
        self.dimension = 0
        self.filenameModelParameter = filenameModelParameter
        self.flagLinearModel = flagLinearModel
        self.minimizationMethod = minimizationMethod
        self.minimizationTolerance = minimizationTolerance
        self.minimizationOptions = self.minimizationOptions0
        for key in minimizationOptions.keys():
            self.minimizationOptions[key] = minimizationOptions[key]
        self.modeInitialPrecisionMatrixMax = modeInitialPrecisionMatrixMax
        self.offsetInitialPrecisionMatrixMax = offsetInitialPrecisionMatrixMax
        # self.constrain = constrain
        # self.mode = mode
        # self.shiftWidth = shiftWidth
        # self.standardizeDescriptor = standardizeDescriptor
        self.widthMeanMonteCarlo = widthMeanMonteCarlo
        self.widthMonteCarlo = widthMonteCarlo
        self.nSampleMonteCarlo = nSampleMonteCarlo
        self.modeAcquisitionFunction = modeAcquisitionFunction
        self.indexTargetPrimary = indexTargetPrimary

        ## Initialization of active variables.
        self.nActiveVariable = 0
        self.mapSerialize = None
        self.mapDeserialize = None

        ## Initialization of data variables.
        self.filenameCovarianceMatrix = filenameCovarianceMatrix
        self.nAvailablePattern = 0
        self.availablePattern = None
        self.covarianceMatrix = None
        self.precisionMatrixMax = None
        self.coef_ = None
        self.targetMax = None
        self.__innerAcquisitionFunction = []
        self.timeCalcPostProb = 0.0

        if((self.nDescriptor < 0) or (self.nTarget < 0)):
            print('Error: {0}, nDescriptor and nTarget must be positive values.'.format(labelFunction))
            sys.exit()

        if((self.nDescriptor > 0) ^ (self.nTarget > 0)):
            print('Error: {0}, nDescriptor and nTarget must be given simultaneously.'.format(labelFunction))
            sys.exit()
        elif((self.nDescriptor > 0) or (self.nTarget > 0)):
            ## Descriptor.
            ## Component of descriptor.
            if (self.flagLinearModel):
                self.dimensionDescriptor = self.nDescriptor + 1
                self.degreePolynomialDescriptor = np.zeros((self.dimensionDescriptor,self.nDescriptor),dtype=int)
                for i in range(1,self.dimensionDescriptor):
                    self.degreePolynomialDescriptor[i,i-1] = 1
            elif (not (degreePolynomialDescriptor is None)):
                if (degreePolynomialDescriptor.shape[1] != self.nDescriptor):
                    print('Error: shape of degreePolynomialDescriptor is illegal')
                    sys.exit()
                self.dimensionDescriptor = len(degreePolynomialDescriptor)
                self.degreePolynomialDescriptor = copy.deepcopy(degreePolynomialDescriptor)
            ## Domain of descriptor.
            if (domainDescriptor is None):
                self.domainDescriptor = np.array([[0.0,1.0] for i in range(self.nDescriptor)])
            else:
                if(len(domainDescriptor) != self.nDescriptor):
                    print('Error: dimension of domain is not same as the number of descriptors.')
                    sys.exit()
                self.domainDescriptor = copy.deepcopy(domainDescriptor)

            ## Target variable.
            ## Component of target variable.
            if (not (degreePolynomialTarget is None)):
                # In the current version, multiple components are not allowed for 
                # target variables, 2022/02/16.
                if (degreePolynomialTarget.shape[1] != self.nTarget):
                    print('Error: shape of degreePolynomialTarget is illegal')
                    sys.exit()
                self.dimensionTarget = len(degreePolynomialTarget)
                self.degreePolynomialTarget = copy.deepcopy(degreePolynomialTarget)
                if(self.dimensionTarget != self.nTarget):
                    print('Error: polynomial for target variable is not yet implemented in the current version.')
                    sys.exit()
                # Set acquisition function pointer.
                self.__innerAcquisitionFunction = [None for i in range(self.dimensionTarget)]
                if (self.modeAcquisitionFunction is None):
                    self.modeAcquisitionFunction = [0 for i in range(self.dimensionTarget)]
                    self.modeAcquisitionFunction[-1] = 1
                elif (len(self.modeAcquisitionFunction) != self.dimensionTarget):
                    print('Error: {0}, modeAcquisitionFunction.'.format(labelFunction))
                    sys.exit()
                self.__setInnerAcquisitionFunction()
                # Set index of primary target variable.
                if (self.indexTargetPrimary < 0):
                    self.indexTargetPrimary = self.dimensionTarget-1

            self.dimension = self.dimensionDescriptor + self.dimensionTarget

            self.activeVariable = np.ones((self.dimensionTarget,self.dimension),dtype=bool)
            for i in range(1,self.dimensionTarget):
                self.activeVariable[i,:self.dimensionDescriptor] = False
            ## Active variable(descriptor and target) for target variable.
            if (not (degreeActiveDescriptor is None)):
                for degreeDescriptor,degreeTarget in degreeActiveDescriptor:
                    found = False
                    for i in range(self.dimensionDescriptor):
                        if(degreeDescriptor == list(self.degreePolynomialDescriptor[i])):
                            idxDescriptor = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of descriptor not found for active variable.'.format(labelFunction))
                        sys.exit()
                    found = False
                    for i in range(self.dimensionTarget):
                        if(degreeTarget == list(self.degreePolynomialTarget[i])):
                            idxTarget = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of target not found for active variable.'.format(labelFunction))
                        sys.exit()
                    self.activeVariable[idxTarget,idxDescriptor] = True
            if (not (degreeInactiveDescriptor is None)):
                for degreeDescriptor,degreeTarget in degreeInactiveDescriptor:
                    found = False
                    for i in range(self.dimensionDescriptor):
                        if(degreeDescriptor == list(self.degreePolynomialDescriptor[i])):
                            idxDescriptor = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of descriptor not found for inactive variable.'.format(labelFunction))
                        sys.exit()
                    found = False
                    for i in range(self.dimensionTarget):
                        if(degreeTarget == list(self.degreePolynomialTarget[i])):
                            idxTarget = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of target not found for inactive variable.'.format(labelFunction))
                        sys.exit()
                    self.activeVariable[idxTarget,idxDescriptor] = False
            if (not (degreeActiveTarget is None)):
                for degreeTarget1,degreeTarget2 in degreeActiveTarget:
                    found = False
                    for i in range(self.dimensionTarget):
                        if(degreeTarget1 == list(self.degreePolynomialTarget[i])):
                            idxTarget1 = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of target1 not found for active variable.'.format(labelFunction))
                        sys.exit()
                    found = False
                    for i in range(self.dimensionTarget):
                        if(degreeTarget2 == list(self.degreePolynomialTarget[i])):
                            idxTarget2 = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of target2 not found for active variable.'.format(labelFunction))
                        sys.exit()
                    self.activeVariable[idxTarget1,self.dimensionDescriptor+idxTarget2] = True
            if (not (degreeInactiveTarget is None)):
                for degreeTarget1,degreeTarget2 in degreeInactiveTarget:
                    found = False
                    for i in range(self.dimensionTarget):
                        if(degreeTarget1 == list(self.degreePolynomialTarget[i])):
                            idxTarget1 = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of target1 not found for inactive variable.'.format(labelFunction))
                        sys.exit()
                    found = False
                    for i in range(self.dimensionTarget):
                        if(degreeTarget2 == list(self.degreePolynomialTarget[i])):
                            idxTarget2 = i
                            found = True
                            break
                    if(not found):
                        print('Error: {0}, degree of target2 not found for inactive variable.'.format(labelFunction))
                        sys.exit()
                    self.activeVariable[idxTarget1,self.dimensionDescriptor+idxTarget2] = False

            # Count number of active variables.
            self.nActiveVariable = 0
            for i in range(self.dimensionTarget):
                for j in range(self.dimensionDescriptor + i + 1):
                    if (self.activeVariable[i,j]):
                        self.nActiveVariable += 1

            ## Calculate mapping from precision matrix to z, i.e., serialization.
            self.__calculateMapSerialize()

            # Initialize parameters for covariance matrix.
            self.nSampleAvailablePattern = np.zeros(self.nAvailablePattern, dtype=int)
            self.availablePattern = np.zeros((self.nAvailablePattern,self.nTarget), dtype=bool)
            self.dimensionAvailablePattern = np.zeros(self.nAvailablePattern, dtype=int)
            self.indexAvailable = [[] for i in range(self.nAvailablePattern)]
            self.indexInavailable = [[] for i in range(self.nAvailablePattern)]
            self.covarianceMatrix = np.zeros((self.nAvailablePattern,self.dimension,self.dimension), dtype=float)
            for idx in range(self.nAvailablePattern):
                self.covarianceMatrix[idx] = np.identity(self.dimension, dtype=float)
            self.precisionMatrixMax = np.identity(self.dimension, dtype=float)
            self.coef_ = np.zeros((self.dimensionTarget,self.dimensionDescriptor), dtype=float)
            self.targetMax = -1e16 * np.ones(self.dimensionTarget, dtype=float)
            self.modeAcquisitionFunction = [0 for i in range(self.dimensionTarget)]
            self.modeAcquisitionFunction[-1] = 1
            self.__innerAcquisitionFunction = [None for i in range(self.dimensionTarget)]
            self.__setInnerAcquisitionFunction()
            if (self.indexTargetPrimary < 0):
                self.indexTargetPrimary = self.dimensionTarget-1
            self.widthMonteCarlo = self.widthMeanMonteCarlo0*np.ones(self.nActiveVariable, dtype=float)
        return

    ### Load model parameters from an xml file.
    def loadModelParameterFromFile(self, \
                                   filenameModelParameter = None):
        labelFunction = 'loadModelParameterFromFile'

        # Filename of model parameters.
        if (not (filenameModelParameter is None)):
            self.filenameModelParameter = filenameModelParameter
        with open(self.filenameModelParameter,'r') as fileobj:
            dom = minidom.parse(fileobj)

        # Header of DOM for model parameters.
        print('{0}: read header of model parameters.'.format(labelFunction))
        header = dom.getElementsByTagName(self.tagNameModelParameter)[0]

        # Number of descriptors.
        print('{0}: read NumberDescriptor.'.format(labelFunction))
        node = header.getElementsByTagName('NumberDescriptor')[0]
        self.nDescriptor = int(node.firstChild.nodeValue.split()[0])
        # flag of linear model.
        node = header.getElementsByTagName('FlagLinearModel')
        if (len(node) > 0):
            if (node[0].firstChild.nodeValue[0] == 'T'):
                self.flagLinearModel = True
        # Components of descriptors.
        print('{0}: read DegreePolynomialDescriptor.'.format(labelFunction))
        if (self.flagLinearModel):
            self.dimensionDescriptor = self.nDescriptor + 1
            self.degreePolynomialDescriptor = np.zeros((self.dimensionDescriptor,self.nDescriptor),dtype=int)
            for i in range(1,self.dimensionDescriptor):
                self.degreePolynomialDescriptor[i,i-1] = 1
        else:
            nodes = header.getElementsByTagName('DegreePolynomialDescriptor')
            degreePolynomialDescriptor = []
            for node in nodes:
                itmpList = [int(stmp) for stmp in node.firstChild.nodeValue.strip().split()]
                if (len(itmpList) != self.nDescriptor):
                    print('Error in {0}: illeagal number of components in degreePolynomialDescriptor.'.format(labelFunction))
                    sys.exit()
                degreePolynomialDescriptor.append(itmpList)
            self.degreePolynomialDescriptor = np.array(degreePolynomialDescriptor)
            self.dimensionDescriptor = self.degreePolynomialDescriptor.shape[0]
        # Domain of descriptors.
        print('{0}: read Domain of descriptor.'.format(labelFunction))
        nodes = header.getElementsByTagName('DomainDescriptor')
        self.domainDescriptor = np.zeros((self.nDescriptor,2),dtype=float)
        for idx in range(self.nDescriptor):
            self.domainDescriptor[idx,1] = 1.0
        for node in nodes:
            idx = int(node.getAttribute('index'))
            self.domainDescriptor[idx,0] = float(node.firstChild.nodeValue.split()[0])
            self.domainDescriptor[idx,1] = float(node.firstChild.nodeValue.split()[1])

        # Number of target variables.
        print('{0}: read NumberTarget.'.format(labelFunction))
        node = header.getElementsByTagName('NumberTarget')[0]
        self.nTarget = int(node.firstChild.nodeValue.split()[0])
        # Components of target variables.
        # In the current version, multiple components are not allowed for 
        # target variables, 2022/02/16.
        print('{0}: read DegreePolynomialTarget.'.format(labelFunction))
        degreePolynomialTarget = []
        nodes = header.getElementsByTagName('DegreePolynomialTarget')
        for node in nodes:
            itmpList = [int(stmp) for stmp in node.firstChild.nodeValue.strip().split()]
            if (len(itmpList) != self.nTarget):
                print('Error in {0}: illeagal number of components in degreePolynomialTarget.'.format(labelFunction))
                sys.exit()
            degreePolynomialTarget.append(itmpList)
        self.degreePolynomialTarget = np.array(degreePolynomialTarget)
        self.dimensionTarget = self.degreePolynomialTarget.shape[0]
        if(self.dimensionTarget != self.nTarget):
            print('Error in {0}: polynomial for target variable is not yet implemented.'.format(labelFunction))
            sys.exit()

        self.dimension = self.dimensionDescriptor + self.dimensionTarget

        # Active variables are initialized, if it is None.
        if (self.activeVariable is None):
            self.activeVariable = np.ones((self.dimensionTarget,self.dimension),dtype=bool)
            for i in range(1,self.dimensionTarget):
                self.activeVariable[i,:self.dimensionDescriptor] = False

        # Active variables for each target variable.
        # The shape of array is, e.g., [[idxTarget,idxVariable], ...].
        # The format of tag is <ActiveDescriptor index=1>2</ActiveDescriptor>, 
        # which corresponds to activating the variable for the target 1 and 
        # variable including descriptors and targets 2.
        print('{0}: read ActiveVariable.'.format(labelFunction))
        nodes = header.getElementsByTagName('ActiveVariable')
        for node in nodes:
            idx1 = int(node.getAttribute('index').split()[0])
            if (idx1 > self.dimensionTarget):
                print('Error in {0}: illeagal target index of ActiveVariable.'.format(labelFunction))
                sys.exit()
            idx2 = int(node.firstChild.nodeValue.split()[0])
            if (idx2 > self.dimension):
                print('Error in {0}: illeagal index of ActiveVariable.'.format(labelFunction))
                sys.exit()
            self.activeVariable[idx1,idx2] = True
            if (idx2 > self.dimensionDescriptor):
                self.activeVariable[idx2 - self.dimensionDescriptor,idx1 + self.dimensionDescriptor] = True

        # Inactive variable for each target.
        print('{0}: read InactiveVariable.'.format(labelFunction))
        nodes = header.getElementsByTagName('InactiveVariable')
        for node in nodes:
            idx1 = int(node.getAttribute('index').split()[0])
            if (idx1 > self.dimensionTarget):
                print('Error in {0}: illeagal target index of InactiveVariable.'.format(labelFunction))
                sys.exit()
            idx2 = int(node.firstChild.nodeValue.split()[0])
            if (idx2 > self.dimension):
                print('Error in {0}: illeagal index of InactiveVariable.'.format(labelFunction))
                sys.exit()
            self.activeVariable[idx1,idx2] = False
            if (idx2 > self.dimensionDescriptor):
                self.activeVariable[idx2 - self.dimensionDescriptor,idx1 + self.dimensionDescriptor] = False

        # Check that all the diagonal components are active.
        for i in range(self.dimensionTarget):
            if (not self.activeVariable[i,i + self.dimensionDescriptor]):
                print('Warning in {0}: diagonal component of activeVariable is set by False at {1:3d}'.format(labelFunction,i))

        # Count number of active variables.
        self.nActiveVariable = 0
        for i in range(self.dimensionTarget):
            for j in range(self.dimensionDescriptor + i + 1):
                if (self.activeVariable[i,j]):
                    self.nActiveVariable += 1

        # Calculate mapping serialization from precision matrix elements to z.
        self.__calculateMapSerialize()

        self.coef_ = np.zeros((self.dimensionTarget,self.dimensionDescriptor), dtype=float)

        # Minimization options.
        node = header.getElementsByTagName('MinimizationMethod')
        if (len(node) > 0):
            nodenode = node[0].getElementsByTagName('Method')
            if(len(nodenode) > 0):
                self.minimizationMethod = nodenode[0].firstChild.nodeValue
            nodenode = node[0].getElementsByTagName('Tolerance')
            if (len(nodenode) > 0):
                self.minimizationTolerance = float(nodenode[0].firstChild.nodeValue.split()[0])
            nodenode = node[0].getElementsByTagName('MaxIter')
            if (len(nodenode) > 0):
                self.minimizationOptions['maxiter'] = int(nodenode[0].firstChild.nodeValue.split()[0])
            nodenode = node[0].getElementsByTagName('FTol')
            if (len(nodenode) > 0):
                self.minimizationOptions['ftol'] = float(nodenode[0].firstChild.nodeValue.split()[0])
            nodenode = node[0].getElementsByTagName('GTol')
            if (len(nodenode) > 0):
                self.minimizationOptions['gtol'] = float(nodenode[0].firstChild.nodeValue.split()[0])
            nodenode = node[0].getElementsByTagName('Eps')
            if (len(nodenode) > 0):
                self.minimizationOptions['eps'] = float(nodenode[0].firstChild.nodeValue.split()[0])
            nodenode = node[0].getElementsByTagName('Disp')
            if (len(nodenode) > 0):
                if (nodenode[0].firstChild.nodeValue[0] == 'T'):
                    self.minimizationOptions['disp'] = True

        # mode for initial precision matrix such that likelihood becomes maximum.
        nodes = header.getElementsByTagName('ModeInitialPrecisionMatrixMax')
        if (len(nodes) > 0):
            self.modeInitialPrecisionMatrixMax = int(nodes[0].firstChild.nodeValue.split()[0])
        else:
            self.modeInitialPrecisionMatrixMax = self.modeInitialPrecisionMatrixMax0

        # offset for initial precision matrix such that likelihood becomes maximum.
        nodes = header.getElementsByTagName('OffsetInitialPrecisionMatrixMax')
        if (len(nodes) > 0):
            self.offsetInitialPrecisionMatrixMax = float(nodes[0].firstChild.nodeValue.split()[0])
        else:
            self.offsetInitialPrecisionMatrixMax = self.offsetInitialPrecisionMatrixMax0

        # # constrained values and columns.
        # nodes = header.getElementsByTagName('Constrain')
        # self.nConstrained = len(nodes)
        # self.columnConstrained = np.zeros(self.nConstrained,dtype=int)
        # self.valueConstrained = np.zeros(self.nConstrained,dtype=float)
        # for i in range(self.nConstrained):
        #     node = nodes[i]
        #     self.columnConstrained[i] = int(node.getAttribute('column'))
        #     self.valueConstrained[i] = float(node.firstChild.nodeValue.split()[0])

        # # random shift for mode = 3.
        # nodes = header.getElementsByTagName('ShiftWidth')
        # if (len(nodes) == 0):
        #     self.shiftWidth = 0.1
        # else:
        #     self.shiftWidth = float(nodes[0].firstChild.nodeValue.split()[0])

        # # standardization.
        # nodes = header.getElementsByTagName('StandardizeDescriptor')
        # if (len(nodes) != 0):
        #     ltmp = list(nodes[0].firstChild.nodeValue.split())
        #     self.standardizeDescriptor = [[int(i),0.0,1.0] for i in ltmp]
        # else:
        #     self.standardizeDescriptor = []
        # nodes = header.getElementsByTagName('StandardizeTarget')
        # if (len(nodes) != 0):
        #     ltmp = list(nodes[0].firstChild.nodeValue.split())
        #     self.standardizeTarget = [[int(i),0.0,1.0] for i in ltmp]
        # else:
        #     self.standardizeTarget = []

        return

    def saveModelParameterToFile(self, \
                                 filenameModelParameter = None):
        labelFunction = 'saveModelParameterToFile'

        if (not (filenameModelParameter is None)):
            self.filenameModelParameter = filenameModelParameter

        impl = minidom.getDOMImplementation()
        doc = impl.createDocument(None,self.tagNameModelParameter,None)
        dom = doc.documentElement

        # Number of descriptors.
        node = doc.createElement('NumberDescriptor')
        textNode = doc.createTextNode(str(self.nDescriptor))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Components of descriptors.
        for idx in range(self.degreePolynomialDescriptor.shape[0]):
            node = doc.createElement('DegreePolynomialDescriptor')
            stmp = ' '.join([str(s) for s in self.degreePolynomialDescriptor[idx]])
            textNode = doc.createTextNode(stmp)
            node.appendChild(textNode)
            dom.appendChild(node)

        for idx in range(self.nDescriptor):
            node = doc.createElement('DomainDescriptor')
            node.setAttribute('index',str(idx))
            stmp = '{0} {1}'.format(self.domainDescriptor[idx,0],self.domainDescriptor[idx,1])
            textNode = doc.createTextNode(stmp)
            node.appendChild(textNode)
            dom.appendChild(node)

        # Number of target variables.
        node = doc.createElement('NumberTarget')
        textNode = doc.createTextNode(str(self.nTarget))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Components of target variables.
        for idx in range(self.degreePolynomialTarget.shape[0]):
            node = doc.createElement('DegreePolynomialTarget')
            stmp = ' '.join([str(iDegree) for iDegree in self.degreePolynomialTarget[idx]])
            textNode = doc.createTextNode(stmp)
            node.appendChild(textNode)
            dom.appendChild(node)

        # Active variables for each target variable.
        for idx1 in range(self.activeVariable.shape[0]):
            for idx2 in range(self.activeVariable.shape[1]):
                if (self.activeVariable[idx1,idx2]):
                    node = doc.createElement('ActiveVariable')
                    node.setAttribute('index',str(idx1))
                    textNode = doc.createTextNode(str(idx2))
                    node.appendChild(textNode)
                    dom.appendChild(node)
                else:
                    node = doc.createElement('InactiveVariable')
                    node.setAttribute('index',str(idx1))
                    textNode = doc.createTextNode(str(idx2))
                    node.appendChild(textNode)
                    dom.appendChild(node)

        # Mode for initial precision matrix such that likelihood becomes maximum.
        node = doc.createElement('ModeInitialPrecisionMatrixMax')
        textNode = doc.createTextNode(str(self.modeInitialPrecisionMatrixMax))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Offset for initial precision matrix such that likelihood becomes maximum.
        node = doc.createElement('OffsetInitialPrecisionMatrixMax')
        textNode = doc.createTextNode(str(self.offsetInitialPrecisionMatrixMax))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Minimization options.
        node = doc.createElement('MinimizationMethod')
        # Method.
        nodenode = doc.createElement('Method')
        textNode = doc.createTextNode(str(self.minimizationMethod))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # Tolerance.
        nodenode = doc.createElement('Tolerance')
        textNode = doc.createTextNode(str(self.minimizationTolerance))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # Maximum iteration.
        nodenode = doc.createElement('MaxIter')
        textNode = doc.createTextNode(str(self.minimizationOptions['maxiter']))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # FTol.
        nodenode = doc.createElement('FTol')
        textNode = doc.createTextNode(str(self.minimizationOptions['ftol']))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # GTol.
        nodenode = doc.createElement('GTol')
        textNode = doc.createTextNode(str(self.minimizationOptions['gtol']))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # Eps.
        nodenode = doc.createElement('Eps')
        textNode = doc.createTextNode(str(self.minimizationOptions['eps']))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # Disp.
        nodenode = doc.createElement('Disp')
        textNode = doc.createTextNode(str(self.minimizationOptions['disp']))
        nodenode.appendChild(textNode)
        node.appendChild(nodenode)
        # Append minimization options to a parent node.
        dom.appendChild(node)

        with open(self.filenameModelParameter,'w') as fileobj:
            dom.writexml(fileobj, newl='\n', addindent='   ')
        return

    ### Load covariance matrix from an xml file.
    def loadCovarianceMatrixFromFile(self, \
                                    filenameCovarianceMatrix = None):
        labelFunction = 'loadCovarianceMatrixFromFile'

        if (filenameCovarianceMatrix):
            self.filenameCovarianceMatrix = filenameCovarianceMatrix
        print('{0}: read covariance matrix from the file: {1}'.format(labelFunction, self.filenameCovarianceMatrix))
        # load covariance matrix elements.
        try:
            fileobj = open(self.filenameCovarianceMatrix,'r')
        except:
            header = None
            self.nAvailablePattern = 0
        else:
            dom = minidom.parse(fileobj)
            fileobj.close()
            # Header of DOM for covariance matrix.
            print('{0}: read header of covariance matrix.'.format(labelFunction))
            headers = dom.getElementsByTagName(self.tagNameCovarianceMatrix)
            if (len(headers) > 1):
                print('Warning in {0}: more than two covariance matrix data block exist.'.format(labelFunction))
            header = headers[0]
            # Number of patterns for available variables.
            node = header.getElementsByTagName('NumberAvailablePattern')[0]
            self.nAvailablePattern = int(node.firstChild.nodeValue)

        # Initialize.
        self.nSampleAvailablePattern = np.zeros(self.nAvailablePattern, dtype=int)
        self.availablePattern = np.zeros((self.nAvailablePattern,self.nTarget), dtype=bool)
        self.dimensionAvailablePattern = np.zeros(self.nAvailablePattern, dtype=int)
        self.indexAvailable = [[] for i in range(self.nAvailablePattern)]
        self.indexInavailable = [[] for i in range(self.nAvailablePattern)]
        self.covarianceMatrix = np.zeros((self.nAvailablePattern,self.dimension,self.dimension), dtype=float)
        for idx in range(self.nAvailablePattern):
            self.covarianceMatrix[idx] = np.identity(self.dimension, dtype=float)
        self.precisionMatrixMax = np.identity(self.dimension, dtype=float)
        self.targetMax = -1e16 * np.ones(self.dimensionTarget, dtype=float)
        self.modeAcquisitionFunction = [0 for i in range(self.dimensionTarget)]
        self.modeAcquisitionFunction[-1] = 1
        self.__innerAcquisitionFunction = [None for i in range(self.dimensionTarget)]
        self.__setInnerAcquisitionFunction()
        if (self.indexTargetPrimary < 0):
            self.indexTargetPrimary = self.dimensionTarget-1
        self.widthMonteCarlo = self.widthMeanMonteCarlo0*np.ones(self.nActiveVariable, dtype=float)

        if (header):
            ## Covariance matrix elements.
            nodes = header.getElementsByTagName('CovarianceMatrix')
            for node in nodes:
                idx = int(node.getAttribute('index'))
                child = node.getElementsByTagName('AvailablePattern')[0]
                sList = child.firstChild.nodeValue.strip().split()
                availablePattern = np.array([flag=='True' for flag in sList])
                if (availablePattern.shape[0] != self.nTarget):
                    print('Error in {0}: illeagal length of availablePattern.'.format(labelFunction))
                    sys.exit()
                self.availablePattern[idx] = availablePattern
                idx1 = self.dimensionDescriptor
                for flag in availablePattern:
                    if (flag):
                        self.indexAvailable[idx].append(idx1)
                    else:
                        self.indexInavailable[idx].append(idx1)
                    idx1 += 1
                self.dimensionAvailablePattern[idx] = availablePattern.sum()
            
                child = node.getElementsByTagName('NumberSampleAvailablePattern')[0]
                self.nSampleAvailablePattern[idx] = int(child.firstChild.nodeValue)
            
                children = node.getElementsByTagName('Element')
                for child in children:
                    irow = int(child.getAttribute('irow'))
                    icol = int(child.getAttribute('icol'))
                    rtmp = float(child.firstChild.nodeValue)
                    self.covarianceMatrix[idx,irow,icol] = rtmp
                    if (irow != icol):
                        self.covarianceMatrix[idx,icol,irow] = rtmp
            ## Maximal likely precision matrix.
            nodes = header.getElementsByTagName('PrecisionMatrixMax')
            if (len(nodes) > 1):
                print('Warning in {0}: more than two maximal likely precision matrix exist.'.format(labelFunction))
            node = nodes[0]
            children = node.getElementsByTagName('Element')
            for child in children:
                irow = int(child.getAttribute('irow'))
                icol = int(child.getAttribute('icol'))
                rtmp = float(child.firstChild.nodeValue)
                self.precisionMatrixMax[irow,icol] = rtmp
                if (irow != icol):
                    self.precisionMatrixMax[icol,irow] = rtmp

            ## Width for Monte Carlo step.
            nodes = header.getElementsByTagName('WidthMonteCarlo')
            if (len(nodes) > 0):
                for node in nodes:
                    if (not node.hasAttributes()):
                        self.widthMeanMonteCarlo = float(node.firstChild.nodeValue.split()[0])
                        break
                self.widthMonteCarlo = self.widthMeanMonteCarlo*np.ones(self.nActiveVariable, dtype=float)
                for node in nodes:
                    if (not node.hasAttributes()):
                        continue
                    idx = int(node.getAttribute('index'))
                    self.widthMonteCarlo[idx] = float(node.firstChild.nodeValue.split()[0])
            ## Number of Monte Carlo samplings.
            nodes = header.getElementsByTagName('NumberSampleMonteCarlo')
            if (len(nodes) > 0):
                self.nSampleMonteCarlo = int(nodes[0].firstChild.nodeValue.split()[0])

            # Acquisition function.
            nodes = header.getElementsByTagName('ModeAcquisitionFunction')
            if (len(nodes) > 0):
                self.modeAcquisitionFunction = [int(stmp) for stmp in nodes[0].firstChild.nodeValue.strip().split()]
            self.__setInnerAcquisitionFunction()

            # Set index of primary target variable.
            nodes = header.getElementsByTagName('IndexTargetPrimary')
            if (len(nodes) > 0):
                self.indexTargetPrimary = int(nodes[0].firstChild.nodeValue.split()[0])
            if (self.indexTargetPrimary < 0):
                self.indexTargetPrimary = self.dimensionTarget-1

            ## Maximum values of target variables.
            nodes = header.getElementsByTagName('TargetMax')
            for node in nodes:
                idx = int(node.getAttribute('index'))
                self.targetMax[idx] = float(node.firstChild.nodeValue.split()[0])
        return

    def saveCovarianceMatrixToFile(self, \
                                  filenameCovarianceMatrix = None):
        labelFunction = 'saveCovarianceMatrixToFile'

        if (filenameCovarianceMatrix):
            self.filenameCovarianceMatrix = filenameCovarianceMatrix

        impl = minidom.getDOMImplementation()
        doc = impl.createDocument(None,self.tagNameCovarianceMatrix,None)
        dom = doc.documentElement

        # Number of patterns for available variables.
        node = doc.createElement('NumberAvailablePattern')
        textNode = doc.createTextNode(str(self.nAvailablePattern))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Covariance matrix.
        for idx in range(self.covarianceMatrix.shape[0]):
            node = doc.createElement('CovarianceMatrix')
            node.setAttribute('index',str(idx))
            # Available patterns.
            child = doc.createElement('AvailablePattern')
            stmp = ' '.join([str(iTarget) for iTarget in self.availablePattern[idx]])
            textNode = doc.createTextNode(stmp)
            child.appendChild(textNode)
            node.appendChild(child)
            # Number of Samples for the available patterns.
            child = doc.createElement('NumberSampleAvailablePattern')
            textNode = doc.createTextNode(str(self.nSampleAvailablePattern[idx]))
            child.appendChild(textNode)
            node.appendChild(child)
            # Matrix elements.
            for irow in range(self.covarianceMatrix.shape[1]):
                for icol in range(irow,self.covarianceMatrix.shape[2]):
                    child = doc.createElement('Element')
                    child.setAttribute('irow',str(irow))
                    child.setAttribute('icol',str(icol))
                    textNode = doc.createTextNode(str(self.covarianceMatrix[idx,irow,icol]))
                    child.appendChild(textNode)
                    node.appendChild(child)
            dom.appendChild(node)
        # Maximal likely precision matrix.
        node = doc.createElement('PrecisionMatrixMax')
        for irow in range(self.precisionMatrixMax.shape[0]):
            for icol in range(irow,self.precisionMatrixMax.shape[1]):
                child = doc.createElement('Element')
                child.setAttribute('irow',str(irow))
                child.setAttribute('icol',str(icol))
                textNode = doc.createTextNode(str(self.precisionMatrixMax[irow,icol]))
                child.appendChild(textNode)
                node.appendChild(child)
        dom.appendChild(node)
        # Coefficients.
        node = doc.createElement('Coefficient')
        for iTarget in range(self.coef_.shape[0]):
            for iDescriptor in range(self.coef_.shape[1]):
                child = doc.createElement('Component')
                child.setAttribute('itarget',str(iTarget))
                child.setAttribute('idescriptor',str(iDescriptor))
                textNode = doc.createTextNode(str(self.coef_[iTarget,iDescriptor]))
                child.appendChild(textNode)
                node.appendChild(child)
        dom.appendChild(node)

        # Width for Monte Carlo step.
        for idx in range(self.widthMonteCarlo.shape[0]):
            node = doc.createElement('WidthMonteCarlo')
            node.setAttribute('index',str(idx))
            textNode = doc.createTextNode(str(self.widthMonteCarlo[idx]))
            node.appendChild(textNode)
            dom.appendChild(node)
        # Number of Monte Carlo samplings.
        node = doc.createElement('NumberSampleMonteCarlo')
        textNode = doc.createTextNode(str(self.nSampleMonteCarlo))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Mode of acquisition functions.
        node = doc.createElement('ModeAcquisitionFunction')
        stmp = ' '.join([str(idx) for idx in self.modeAcquisitionFunction])
        textNode = doc.createTextNode(stmp)
        node.appendChild(textNode)
        dom.appendChild(node)

        # Index of primary target variable.
        node = doc.createElement('IndexTargetPrimary')
        textNode = doc.createTextNode(str(self.indexTargetPrimary))
        node.appendChild(textNode)
        dom.appendChild(node)

        # Maximum values of target variables.
        for idx in range(self.dimensionTarget):
            node = doc.createElement('TargetMax')
            node.setAttribute('index',str(idx))
            textNode = doc.createTextNode(str(self.targetMax[idx]))
            node.appendChild(textNode)
            dom.appendChild(node)

        with open(self.filenameCovarianceMatrix,'w') as fileobj:
            dom.writexml(fileobj, newl='\n', addindent='   ')
        return

    ### Reset maximal likely precision matrix.
    def resetPrecisionMatrixMax(self):
        labelFunction = 'resetPrecisionMatrixMax'

        if (self.modeInitialPrecisionMatrixMax == 0):
            self.precisionMatrixMax = np.identity(self.dimension, dtype=float)
        elif (self.modeInitialPrecisionMatrixMax == 1):
            covarianceMatrixAverage = np.zeros((self.dimension,self.dimension), dtype=float)
            for idx1 in range(self.dimensionDescriptor):
                for idx2 in range(idx1+1):
                    nSample = 0
                    for idx in range(self.nAvailablePattern):
                        covarianceMatrixAverage[idx1,idx2] += self.covarianceMatrix[idx,idx1,idx2] * self.nSampleAvailablePattern[idx]
                        nSample += self.nSampleAvailablePattern[idx]
                    covarianceMatrixAverage[idx1,idx2] /= nSample
            for idx1 in range(self.dimensionDescriptor, self.dimension):
                for idx2 in range(self.dimensionDescriptor):
                    nSample = 0
                    for idx in range(self.nAvailablePattern):
                        if (not self.availablePattern[idx,idx1-self.dimensionDescriptor]):
                            continue
                        covarianceMatrixAverage[idx1,idx2] += self.covarianceMatrix[idx,idx1,idx2] * self.nSampleAvailablePattern[idx]
                        nSample += self.nSampleAvailablePattern[idx]
                    covarianceMatrixAverage[idx1,idx2] /= max(nSample,1)
                for idx2 in range(self.dimensionDescriptor, idx1+1):
                    nSample = 0
                    for idx in range(self.nAvailablePattern):
                        if (not self.availablePattern[idx,idx1-self.dimensionDescriptor]):
                            continue
                        if (not self.availablePattern[idx,idx2-self.dimensionDescriptor]):
                            continue
                        covarianceMatrixAverage[idx1,idx2] += self.covarianceMatrix[idx,idx1,idx2] * self.nSampleAvailablePattern[idx]
                        nSample += self.nSampleAvailablePattern[idx]
                    covarianceMatrixAverage[idx1,idx2] /= max(nSample,1)
            for idx1 in range(self.dimension):
                for idx2 in range(idx1):
                    covarianceMatrixAverage[idx2,idx1] = covarianceMatrixAverage[idx1,idx2]
            self.precisionMatrixMax = inv(covarianceMatrixAverage)
            for idx1 in range(self.dimensionTarget):
                for idx2 in range(self.dimensionDescriptor + idx1):
                    if (not self.activeVariable[idx1,idx2]):
                        self.precisionMatrixMax[self.dimensionDescriptor + idx1,idx2] = 0.0
                        self.precisionMatrixMax[idx2,self.dimensionDescriptor + idx1] = 0.0
            for idx1 in range(self.dimension):
                self.precisionMatrixMax[idx1,idx1] += self.offsetInitialPrecisionMatrixMax
        return

    ###
    def resetCovarianceMatrix(self):
        self.nAvailablePattern = 0
        self.nSampleAvailablePattern = np.zeros(self.nAvailablePattern, dtype=int)
        self.availablePattern = np.zeros((self.nAvailablePattern,self.nTarget), dtype=bool)
        self.dimensionAvailablePattern = np.zeros(self.nAvailablePattern, dtype=int)
        self.indexAvailable = [[] for i in range(self.nAvailablePattern)]
        self.indexInavailable = [[] for i in range(self.nAvailablePattern)]
        self.covarianceMatrix = np.zeros((self.nAvailablePattern,self.dimension,self.dimension), dtype=float)
        for idx in range(self.nAvailablePattern):
            self.covarianceMatrix[idx] = np.identity(self.dimension, dtype=float)
        return

    ###
    def fit(self, X, Y):
        labelFunction = 'fit'
        if(X.ndim != 2):
            print('Error in {0}: dimension of descriptor sample is not 2, but {1}.'.format(labelFunction,X.ndim))
            sys.exit()
        if(Y.ndim != 2):
            print('Error in {0}: dimension of target sample is not 2, but {1}.'.format(labelFunction,Y.ndim))
            sys.exit()

        self.resetCovarianceMatrix()
        for i in range(X.shape[0]):
            self.updateCovarianceMatrix(X[i],Y[i])
        self.resetPrecisionMatrixMax()
        self.findMaximumPosteriorProbability()
        coef = self.calculateLinearRegressionParameter()
        return

    ### Find the Lambda such that the maximum posterior probability.
    def findMaximumPosteriorProbability(self):
        labelFunction = 'findMaximumPosteriorProbability'

        ## Maximize the likelihood. 
        ## __minimization_func is, roughly speaking, (-1) * likelihood.
        z0 = self.__serializePrecisionMatrix(self.precisionMatrixMax)
        if (self.useBasinHopping):
            ret = basinhopping(self.__minimization_func,z0, \
                               niter=self.iterationBasinHopping, \
                               # minimizer_kwargs={'method':self.minimizationMethod, \
                               #                   'tol':self.minimizationTolerance, \
                               #                   **self.minimizationOptions}, \
                               minimizer_kwargs={'method':self.minimizationMethod, \
                                                 'tol':self.minimizationTolerance}, \
                               disp=self.minimizationOptions['disp'])
        else:
            ret = minimize(self.__minimization_func,z0, \
                           method=self.minimizationMethod, \
                           tol=self.minimizationTolerance, \
                           options=self.minimizationOptions)
        self.precisionMatrixMax = self.__deserializePrecisionMatrix(ret.x)
        if (not ret.success):
            print('Warning in {0}: maximization for posterior probability is not converged.'.format(labelFunction))
        return

    ###---------------------------------------------------------------------------
    # The model which corresponds maximum posterior probability are used 
    # to predict the target variable for the given descriptor, x.
    ###---------------------------------------------------------------------------
    def predict(self, x, \
                precisionMatrix = None):
        labelFunction = 'predict'

        if(isinstance(x, list) or isinstance(x, np.ndarray)):
            x0 = copy.deepcopy(x)
            if(isinstance(x0, list)):
                x0 = np.array(x0)
            if(x0.ndim == 1):
                if(self.nDescriptor == 1):
                    # x0 = [value, value, ...].
                    dtmp = x0.reshape((-1,1))
                else: # not self.nDescriptor == 1.
                    if(x0.shape[0] == self.nDescriptor):
                        dtmp = np.array([x0])
                    else:
                        print('Error: illeagal shape in {0}.'.format(labelFunction))
                        sys.exit()
            elif(x0.ndim == 2):
                dtmp = x0
            else:
                print('Error: illeagal dimension in {0}.'.format(labelFunction))
                sys.exit()
        else:
            dtmp = np.array([[x]]) # assume x as a scalar.
        # if (len(self.standardizeDescriptor) != 0):
        #     for i in range(len(self.standardizeDescriptor)):
        #         dtmp[self.standardizeDescriptor[i][0]] -= self.standardizeDescriptor[i][1]
        #         dtmp[self.standardizeDescriptor[i][0]] /= self.standardizeDescriptor[i][2]
        xtmp = np.ones((dtmp.shape[0],self.dimensionDescriptor),dtype=float)
        for idx1 in range(self.dimensionDescriptor):
            for idx2 in range(self.degreePolynomialDescriptor[idx1].shape[0]):
                xtmp[:,idx1] *= dtmp[:,idx2]**self.degreePolynomialDescriptor[idx1,idx2]

        if (precisionMatrix is None):
            precisionMatrix = self.precisionMatrixMax

        precisionMatrixTargetInv = inv(precisionMatrix[self.dimensionDescriptor:,self.dimensionDescriptor:])
        vectmp = np.matmul(precisionMatrix[self.dimensionDescriptor:,:self.dimensionDescriptor], xtmp.T)
        y = -np.matmul(precisionMatrixTargetInv,vectmp).T

        # if (len(self.standardizeTarget) != 0):
        #     for i in range(len(self.standardizeTarget)):
        #         y[self.standardizeTarget[i][0]] *= self.standardizeTarget[i][2]
        #         y[self.standardizeTarget[i][0]] += self.standardizeTarget[i][1]

        if(isinstance(x, list) or isinstance(x, np.ndarray)):
            if(x0.ndim == 1):
                if(self.nDescriptor != 1):
                    y = y.reshape(-1)
        else:
            y = y.reshape(-1)
        return y

    def __calculateMapSerialize(self):
        labelFunction = '__calculateMapSerialize'

        self.mapSerialize = -np.ones((self.dimensionTarget,self.dimension),dtype=int)
        self.mapDeserialize = []
        idx = 0
        for j1 in range(self.dimensionTarget):
            for j2 in range(self.dimensionDescriptor + j1 + 1):
                if (self.activeVariable[j1,j2]):
                    self.mapSerialize[j1,j2] = idx
                    self.mapDeserialize.append([j1,j2])
                    idx += 1
        for j1 in range(self.dimensionTarget):
            for j2 in range(j1 + 1, self.dimensionTarget):
                self.mapSerialize[j1,j2 + self.dimensionDescriptor] = self.mapSerialize[j2,j1 + self.dimensionDescriptor]
        self.mapDeserialize = np.array(self.mapDeserialize)
        return

    def __deserializePrecisionMatrix(self,z):
        labelFunction = '__deserializePrecisionMatrix'

        precisionMatrix = np.zeros((self.dimension,self.dimension), dtype=float)
        for j1 in range(self.dimensionTarget):
            for j2 in range(self.dimensionDescriptor + j1):
                if (self.mapSerialize[j1,j2] < 0):
                    continue
                if (not self.activeVariable[j1,j2]):
                    continue
                precisionMatrix[j1 + self.dimensionDescriptor,j2] = z[self.mapSerialize[j1,j2]]
                precisionMatrix[j2,j1 + self.dimensionDescriptor] = z[self.mapSerialize[j1,j2]]
            precisionMatrix[j1 + self.dimensionDescriptor,j1 + self.dimensionDescriptor] = z[self.mapSerialize[j1,j1 + self.dimensionDescriptor]]
        return precisionMatrix

    def __serializePrecisionMatrix(self, precisionMatrix):
        labelFunction = '__serializePrecisionMatrix'

        z = np.zeros(self.nActiveVariable,dtype=float)
        for j1 in range(self.dimensionTarget):
            for j2 in range(self.dimensionDescriptor + j1 + 1):
                if (self.activeVariable[j1,j2]):
                    z[self.mapSerialize[j1,j2]] = precisionMatrix[j1 + self.dimensionDescriptor,j2]
        return z

    def __minimization_func(self,z):
        labelFunction = '__minimization_func'

        timetmp = time.time()
        precisionMatrix = self.__deserializePrecisionMatrix(z)
        value = -self.calculatePosteriorProbability(precisionMatrix)
        self.timeCalcPostProb += time.time()-timetmp
        return value

    ### Logarithm of posterior probability.
    def calculatePosteriorProbability(self, precisionMatrix):
        labelFunction = 'calculatePosteriorProbability'

        ## To be implemented to use any prior probability distribution in the future.
        priorProbability = 1.0
        return self.calculateLikelihood(precisionMatrix) + np.log(priorProbability)

    def calculateLikelihood(self, precisionMatrix):
        labelFunction = 'calculateLikelihood'

        value = 0.0
        for idx in range(self.nAvailablePattern):
            precisionMatrixAvailable,detPrecisionMatrixAvailable = \
                self.__calculatePrecisionMatrixAvailable(idx, precisionMatrix)
            rtmp1 = 0.0
            for j1 in range(self.dimensionDescriptor):
                for j2 in range(j1):
                    rtmp1 += 2 * precisionMatrixAvailable[j1,j2] * self.covarianceMatrix[idx,j2,j1]
                rtmp1 += precisionMatrixAvailable[j1,j1] * self.covarianceMatrix[idx,j1,j1]
            for j1 in range(self.dimensionDescriptor,self.dimension):
                if (not self.availablePattern[idx,j1-self.dimensionDescriptor]):
                    continue
                for j2 in range(self.dimensionDescriptor):
                    rtmp1 += 2 * precisionMatrixAvailable[j1,j2] * self.covarianceMatrix[idx,j2,j1]
                for j2 in range(self.dimensionDescriptor,j1):
                    if (not self.availablePattern[idx,j2-self.dimensionDescriptor]):
                        continue
                    rtmp1 += 2 * precisionMatrixAvailable[j1,j2] * self.covarianceMatrix[idx,j2,j1]
                rtmp1 += precisionMatrixAvailable[j1,j1] * self.covarianceMatrix[idx,j1,j1]
            value += 0.5 * self.nSampleAvailablePattern[idx] \
                * (-self.dimensionAvailablePattern[idx] * np.log(2*np.pi) + np.log(detPrecisionMatrixAvailable) - rtmp1)
        return value

    ### Calculate reduced precision matrix for an available pattern.
    def __calculatePrecisionMatrixAvailable(self, idx, precisionMatrix):
        labelFunction = '__calculatePrecisionMatrixAvailable'

        ## G1: Integrating missing variables.
        if (self.dimensionTarget == self.dimensionAvailablePattern[idx]): # For no missing data.
            ## Precision matrix for available variables * available variables.
            ## Lambda'_y'y' = Lambda_y'y' - Lambda_y'bar{y} * (Lambda_bar{y}bar{y}^(-1) * Lambda_bar{y}y').
            mattmp1 = np.zeros((self.dimensionAvailablePattern[idx],self.dimensionAvailablePattern[idx]),dtype=float)
            for j1 in range(self.dimensionAvailablePattern[idx]): # Available target.
                for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                    mattmp1[j1,j2] = precisionMatrix[self.indexAvailable[idx][j1],self.indexAvailable[idx][j2]]
            ## Precision matrix for descriptor * available variables.
            mattmp2 = np.zeros((self.dimensionDescriptor,self.dimensionAvailablePattern[idx]),dtype=float)
            for j1 in range(self.dimensionDescriptor): # Descriptor.
                for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                    mattmp2[j1,j2] = precisionMatrix[j1,self.indexAvailable[idx][j2]]
        else: # For missing data.
            ## Precision matrix for inavailable variables * inavailable variables.
            ## mattmp4: Lambda_bar{y}bar{y}^(-1).
            mattmp4 = np.zeros((self.dimensionTarget-self.dimensionAvailablePattern[idx], \
                                self.dimensionTarget-self.dimensionAvailablePattern[idx]),dtype=float)
            for j1 in range(self.dimensionTarget-self.dimensionAvailablePattern[idx]): # Inavailable target.
                for j2 in range(self.dimensionTarget-self.dimensionAvailablePattern[idx]): # Inavailable target.
                    mattmp4[j1,j2] = precisionMatrix[self.indexInavailable[idx][j1],self.indexInavailable[idx][j2]]
            mattmp4 = inv(mattmp4)
            ## Precision matrix for inavailable variables * available variables.
            ## mattmp5: Lambda_bar{y}y'.
            mattmp5 = np.zeros((self.dimensionTarget-self.dimensionAvailablePattern[idx], \
                                self.dimensionAvailablePattern[idx]),dtype=float)
            for j1 in range(self.dimensionTarget-self.dimensionAvailablePattern[idx]): # Inavailable target.
                for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                    mattmp5[j1,j2] = precisionMatrix[self.indexInavailable[idx][j1],self.indexAvailable[idx][j2]]
            ## mattmp6: Lambda_bar{y}bar{y}^(-1) * Lambda_bar{y}y'.
            ## mattmp6(dimInavail,dimAvail).
            mattmp6 = np.matmul(mattmp4,mattmp5)
            ## Precision matrix for available variables * available variables.
            ## mattmp1: Lambda'_y'y' = Lambda_y'y' - Lambda_y'bar{y} * Lambda_bar{y}bar{y}^(-1) * Lambda_bar{y}y'.
            mattmp1 = -np.matmul(mattmp5.T,mattmp6)
            for j1 in range(self.dimensionAvailablePattern[idx]): # Available target.
                for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                    mattmp1[j1,j2] += precisionMatrix[self.indexAvailable[idx][j1],self.indexAvailable[idx][j2]]

            ## Precision matrix for descriptor * available variables.
            mattmp7 = np.zeros((self.dimensionDescriptor, \
                                 self.dimensionTarget-self.dimensionAvailablePattern[idx]),dtype=float)
            for j1 in range(self.dimensionDescriptor): # Descriptor.
                for j2 in range(self.dimensionTarget-self.dimensionAvailablePattern[idx]): # Inavailable target.
                    mattmp7[j1,j2] = precisionMatrix[j1,self.indexInavailable[idx][j2]]
            mattmp2 = -np.matmul(mattmp7,mattmp6)
            for j1 in range(self.dimensionDescriptor): # Descriptor.
                for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                    mattmp2[j1,j2] += precisionMatrix[j1,self.indexAvailable[idx][j2]]

        ## Determinant of precision matrix for available variables.
        detPrecisionMatrixAvailable = det(mattmp1)

        ## G2: Conditional probability for likelihood calculation.
        precisionMatrixAvailable = np.zeros((self.dimension,self.dimension), dtype=float)
        ## Lambda0_xy'.
        for j1 in range(self.dimensionDescriptor): # Descriptor.
            for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                precisionMatrixAvailable[j1,self.indexAvailable[idx][j2]] = mattmp2[j1,j2]
                precisionMatrixAvailable[self.indexAvailable[idx][j2],j1] = mattmp2[j1,j2]
        ## Lambda0_y'y'.
        for j1 in range(self.dimensionAvailablePattern[idx]): # Available target.
            for j2 in range(self.dimensionAvailablePattern[idx]): # Available target.
                precisionMatrixAvailable[self.indexAvailable[idx][j1],self.indexAvailable[idx][j2]] = mattmp1[j1,j2]
        mattmp1 = inv(mattmp1)
        mattmp3 = np.matmul(mattmp2,np.matmul(mattmp1,mattmp2.T))
        for j1 in range(self.dimensionDescriptor): # Descriptor.
            for j2 in range(self.dimensionDescriptor): # Descriptor.
                precisionMatrixAvailable[j1,j2] = mattmp3[j1,j2]
        ## Return self.dimension * self.dimension precision matrix, and 
        ## determinant of an available part in precision matrix.
        return precisionMatrixAvailable, detPrecisionMatrixAvailable

    ### The parameters of corresponding linear regression model using the most likely precision matrix.
    def calculateLinearRegressionParameter(self, \
                                           precisionMatrix = None):
        labelFunction = 'calculateLinearRegressionParameter'

        if (precisionMatrix is None):
            precisionMatrix = self.precisionMatrixMax

        # projection.
        mattmp = np.zeros((self.dimensionTarget,self.dimensionTarget),dtype=float)
        for j1 in range(self.dimensionTarget):
            jj1 = self.dimensionDescriptor+j1
            for j2 in range(self.dimensionTarget):
                jj2 = self.dimensionDescriptor+j2
                mattmp[j1,j2] = precisionMatrix[jj1,jj2]
        mattmp = inv(mattmp)
        self.coef_ = -np.matmul(mattmp,precisionMatrix[self.dimensionDescriptor:,:self.dimensionDescriptor])
        return copy.deepcopy(self.coef_)

    ###---------------------------------------------------------------------------
    # Updating the covariance matrix using new data.
    # The available pattern is checked and the covariance matrix will be added when 
    # no matched pattern exists.
    ###---------------------------------------------------------------------------
    def updateCovarianceMatrix(self,x,y):
        labelFunction = 'updateCovarianceMatrix'

        for idx in range(self.dimensionTarget):
            if (np.isnan(y[idx])):
                continue
            if (y[idx] > self.targetMax[idx]):
                self.targetMax[idx] = y[idx]

        xVariable = np.ones(self.dimensionDescriptor, dtype=float)
        for idx1 in range(self.dimensionDescriptor):
            for idx2 in range(self.degreePolynomialDescriptor[idx1].shape[0]):
                xVariable[idx1] *= x[idx2]**self.degreePolynomialDescriptor[idx1,idx2]
        yVariable = np.ones(self.dimensionTarget, dtype=float)
        for idx1 in range(self.dimensionTarget):
            for idx2 in range(self.degreePolynomialTarget[idx1].shape[0]):
                yVariable[idx1] *= y[idx2]**self.degreePolynomialTarget[idx1,idx2]
        xyVariable = np.concatenate([xVariable,yVariable])

        covarianceMatrixTmp = np.identity(self.dimension,dtype=float)
        availablePatternTmp = np.ones(self.dimensionTarget,dtype=bool)
        for j1 in range(self.dimensionDescriptor):
            for j2 in range(j1):
                covarianceMatrixTmp[j1,j2] = xyVariable[j1]*xyVariable[j2]
                covarianceMatrixTmp[j2,j1] = covarianceMatrixTmp[j1,j2]
            covarianceMatrixTmp[j1,j1] = xyVariable[j1]**2
        for j1 in range(self.dimensionDescriptor,self.dimension):
            if (np.isnan(xyVariable[j1])):
                availablePatternTmp[j1-self.dimensionDescriptor] = False
                continue
            for j2 in range(self.dimensionDescriptor):
                covarianceMatrixTmp[j1,j2] = xyVariable[j1]*xyVariable[j2]
                covarianceMatrixTmp[j2,j1] = covarianceMatrixTmp[j1,j2]
            for j2 in range(self.dimensionDescriptor,j1):
                if (np.isnan(xyVariable[j2])):
                    continue
                covarianceMatrixTmp[j1,j2] = xyVariable[j1]*xyVariable[j2]
                covarianceMatrixTmp[j2,j1] = covarianceMatrixTmp[j1,j2]
            covarianceMatrixTmp[j1,j1] = xyVariable[j1]**2

        found = False
        if(self.nAvailablePattern == 0):
            self.nSampleAvailablePattern = np.empty((0,1), dtype=int)
            self.availablePattern = np.empty((0, availablePatternTmp.shape[0]), dtype=bool)
        else:
            for idx in range(self.nAvailablePattern):
                if (all(~availablePatternTmp^self.availablePattern[idx])):
                    indexAvailablePattern = idx
                    found = True
                    break
        if (found):
            self.nSampleAvailablePattern[indexAvailablePattern] += 1
            nSampleTmp = self.nSampleAvailablePattern[indexAvailablePattern]
            self.covarianceMatrix[indexAvailablePattern] = covarianceMatrixTmp / float(nSampleTmp) \
                + self.covarianceMatrix[indexAvailablePattern] * float(nSampleTmp-1)/float(nSampleTmp)
        else:
            indexAvailablePattern = self.nAvailablePattern
            self.nAvailablePattern += 1
            self.nSampleAvailablePattern = np.append(self.nSampleAvailablePattern, 1)
            self.availablePattern = np.append(self.availablePattern, [availablePatternTmp], axis=0)
            self.indexAvailable.append([])
            self.indexInavailable.append([])
            idx1 = self.dimensionDescriptor
            for flag in availablePatternTmp:
                if (flag):
                    self.indexAvailable[indexAvailablePattern].append(idx1)
                else:
                    self.indexInavailable[indexAvailablePattern].append(idx1)
                idx1 += 1
            self.dimensionAvailablePattern = np.append(self.dimensionAvailablePattern, availablePatternTmp.sum())
            self.covarianceMatrix = np.append(self.covarianceMatrix, [covarianceMatrixTmp], axis=0)
        return

    ### Transition of precision matrix.
    def transitionPrecisionMatrix(self, precisionMatrix):
        labelFunction = 'transitionPrecisionMatrix'
        # Posterior probability.
        postProb = self.calculatePosteriorProbability(precisionMatrix)
        precisionMatrixNext = copy.deepcopy(precisionMatrix)
        idxList = list(range(self.nActiveVariable))
        np.random.shuffle(idxList)
        flagChanged = False
        for iComponent in idxList:
            j1, j2 = self.mapDeserialize[iComponent]
            j1 += self.dimensionDescriptor
            # Next candidate.
            self.deviatePrecisionMatrix(precisionMatrixNext, iComponent)
            postProbNext = self.calculatePosteriorProbability(precisionMatrixNext)
            dp = postProbNext - postProb
            if (np.random.rand() < np.exp(dp)):
                precisionMatrix[j1,j2] = precisionMatrixNext[j1,j2]
                precisionMatrix[j2,j1] = precisionMatrix[j1,j2]
                postProb = postProbNext
                flagChanged = True
            else:
                precisionMatrixNext[j1,j2] = precisionMatrix[j1,j2]
                precisionMatrixNext[j2,j1] = precisionMatrix[j1,j2]
            if (flagDebugPostProb):
                with open(filenameDebugPostProb, 'a') as fileObj:
                    print('  {0:8d}  {1:.8e}  {2:.8e}'.format(iComponent, postProb, postProbNext), file=fileObj)
        return flagChanged

    ### Deviating precision matrix from a given precision matrix.
    def deviatePrecisionMatrix(self, precisionMatrix, iComponent):
        labelFunction = 'deviatePrecisionMatrix'
        j1, j2 = self.mapDeserialize[iComponent]
        j1 += self.dimensionDescriptor
        dz = (np.random.rand() - 0.5) * self.widthMonteCarlo[iComponent]
        precisionMatrix[j1,j2] = precisionMatrix[j1,j2] + dz
        precisionMatrix[j2,j1] = precisionMatrix[j1,j2]
        return

    ###---------------------------------------------------------------------------
    # Calculate values of acquisition function for a given descriptor value.
    # The acquisition function indicates benefit to choose the point.
    # x must be 2-dimensional array: x[:,:] or x[:][:]
    ###---------------------------------------------------------------------------
    def acquisitionFunction(self, x):
        labelFunction = 'acquisitionFunction'
        precisionMatrix = copy.deepcopy(self.precisionMatrixMax)
        zMean = self.__serializePrecisionMatrix(precisionMatrix)
        z = copy.deepcopy(zMean)
        coefSample = []
        # zSample = []
        dzSq = np.zeros(self.nActiveVariable, dtype=float)
        for iSample in range(self.nSampleMonteCarlo):
            flagChanged = self.transitionPrecisionMatrix(precisionMatrix)
            if (flagChanged):
                z = self.__serializePrecisionMatrix(precisionMatrix)
            else:
                self.widthMonteCarlo = 0.9 * self.widthMonteCarlo
            coef_ = self.calculateLinearRegressionParameter(precisionMatrix)
            coefSample.append(coef_)
            # zSample.append(list(z))
            dzSq += (z - zMean)**2
            if (((iSample+1) % 20) == 0):
                self.widthMonteCarlo = 0.75*self.widthMonteCarlo + 0.25*np.sqrt(dzSq/(iSample+1))

        if(isinstance(x, list)):
            x0 = np.array(x)
        else:
            x0 = x

        value = np.zeros(self.dimensionTarget, dtype=float)
        values = []
        ySample = np.zeros((self.dimensionTarget,self.nSampleMonteCarlo), dtype=float)
        for ix in range(len(x)):
            xtmp = np.ones(self.dimensionDescriptor,dtype=float)
            for idx1 in range(self.dimensionDescriptor):
                for idx2 in range(self.degreePolynomialDescriptor[idx1].shape[0]):
                    xtmp[idx1] *= x0[ix,idx2]**self.degreePolynomialDescriptor[idx1,idx2]
            for iSample in range(self.nSampleMonteCarlo):
                y = np.matmul(coefSample[iSample],xtmp)
                # precisionMatrix = self.__deserializePrecisionMatrix(zSample[iSample])
                # y = self.predict(x[ix], precisionMatrix)
                for idx in range(self.dimensionTarget):
                    ySample[idx,iSample] = y[idx]
            for idx in range(self.dimensionTarget):
                value[idx] = self.__innerAcquisitionFunction[idx](ySample, idx)
            values.append(list(value))
        return np.array(values)

        # y = self.predict(x, precisionMatrix)
        # ySample = np.zeros((self.dimensionTarget,self.nSampleMonteCarlo), dtype=float)
        # dzSq = np.zeros(self.nActiveVariable, dtype=float)
        # ## Debug.
        # if(flagDebugLambda):
        #     fileObj1 = open(filenameDebugLambda, 'a')
        #     print('## x=', end='', file=fileObj1)
        #     for i in range(len(x)):
        #         print('  {0:.8f}'.format(x[i]), end='', file=fileObj1)
        #     print('', file=fileObj1)
        # if(flagDebugWidthMonteCarlo):
        #     fileObj2 = open(filenameDebugWidthMonteCarlo, 'a')
        #     print('## x=', end='', file=fileObj2)
        #     for i in range(len(x)):
        #         print('  {0:.8f}'.format(x[i]), end='', file=fileObj2)
        #     print('', file=fileObj2)
        # for iSample in range(self.nSampleMonteCarlo):
        #     ## Debug.
        #     if (flagDebugPostProb):
        #         with open(filenameDebugPostProb, 'a') as fileObj3:
        #             print('iSample: {0}'.format(iSample), file=fileObj3)
        #     flagChanged = self.transitionPrecisionMatrix(precisionMatrix)
        #     if (flagChanged):
        #         z = self.__serializePrecisionMatrix(precisionMatrix)
        #     else:
        #         self.widthMonteCarlo = 0.9 * self.widthMonteCarlo
        #     dzSq += (z - zMean)**2
        #     if (((iSample+1) % 20) == 0):
        #         self.widthMonteCarlo = 0.75*self.widthMonteCarlo + 0.25*np.sqrt(dzSq/(iSample+1))
        #     y = self.predict(x, precisionMatrix)
        #     for idx in range(self.dimensionTarget):
        #         ySample[idx,iSample] = y[idx]
        #     ## Debug.
        #     if(flagDebugLambda):
        #         print('{0:8d}'.format(iSample), end='', file=fileObj1)
        #         for i in range(self.nActiveVariable):
        #             print('  {0:20.8f}'.format(z[i]), end='', file=fileObj1)
        #         print('', file=fileObj1)
        #     if(flagDebugWidthMonteCarlo):
        #         print('{0:8d}'.format(iSample), end='', file=fileObj2)
        #         for i in range(self.nActiveVariable):
        #             print('  {0:20.8f}'.format(self.widthMonteCarlo[i]), end='', file=fileObj2)
        #         print('', file=fileObj2)
        # value = np.zeros(self.dimensionTarget, dtype=float)
        # for idx in range(self.dimensionTarget):
        #     value[idx] = self.__innerAcquisitionFunction[idx](ySample, idx)
        # ## Debug.
        # if(flagDebugLambda):
        #     fileObj1.close()
        # if(flagDebugWidthMonteCarlo):
        #     fileObj2.close()
        # return value

    ### Set acquisition function pointer.
    def __setInnerAcquisitionFunction(self):
        labelFunction = '__setInnerAcquisitionFunction'
        for idx in range(len(self.modeAcquisitionFunction)):
            if (self.modeAcquisitionFunction[idx] == 0):
                self.__innerAcquisitionFunction[idx] = self.__innerStandardDeviation
            elif (self.modeAcquisitionFunction[idx] == 1):
                self.__innerAcquisitionFunction[idx] = self.__innerExpectedImprovement
            elif (self.modeAcquisitionFunction[idx] == 2):
                self.__innerAcquisitionFunction[idx] = self.__innerCorrelationTargetPrimary
            else:
                print('Warning {0}: unknown acquisition function, set from {1} to 0.'.format(labelFunction, self.modeAcquisitionFunction[idx]))
                self.__innerAcquisitionFunction[idx] = self.__innerStandardDeviation
        return

    ###---------------------------------------------------------------------------
    # Expected improvement for acquisition function. 
    # The form of data is assumed to be (dimensionTarget, nSampleMonteCarlo).
    ###---------------------------------------------------------------------------
    def __innerExpectedImprovement(self, ySample, idx):
        return np.maximum(ySample[idx,:] - self.targetMax[idx], 0.0).mean()

    ###---------------------------------------------------------------------------
    # Standard deviation for acquisition function. 
    # The form of data is assumed to be (dimensionTarget, nSampleMonteCarlo).
    ###---------------------------------------------------------------------------
    def __innerStandardDeviation(self, ySample, idx):
        return np.sqrt((ySample[idx,:]**2).mean() - (ySample[idx,:].mean())**2)

    ###---------------------------------------------------------------------------
    # Correlation with the primary target variable. 
    # Used for acquisition function. 
    # The form of data is assumed to be (dimensionTarget, nSampleMonteCarlo).
    ###---------------------------------------------------------------------------
    def __innerCorrelationTargetPrimary(self, ySample, idx):
        return (ySample[idx,:]*ySample[self.indexTargetPrimary,:]).mean() - ySample[idx,:].mean()*ySample[self.indexTargetPrimary,:].mean()

    ### Getter for domain of descriptor.
    def getDomainDescriptor(self):
        return copy.deepcopy(self.domainDescriptor)

    ### Setter for domain of descriptor.
    def setDomainDescriptor(self,domainDescriptor):
        self.domainDescriptor = domainDescriptor
        return

    ### Getter for number of descriptors.
    def getNDescriptor(self):
        return self.nDescriptor

    ### Getter for number of target variables.
    def getNTarget(self):
        return self.nTarget

    ### Getter for dimension of descriptors.
    def getDimensionDescriptor(self):
        return self.dimensionDescriptor

    ### Getter for dimension of target variables.
    def getDimensionTarget(self):
        return self.dimensionTarget

    # def getStandardizeDescriptor(self):
    #     return copy.deepcopy(self.standardizeDescriptor)

    # def getStandardizeTarget(self):
    #     return copy.deepcopy(self.standardizeTarget)

    ### Getter for degree of polynomials of descriptors.
    def getDegreePolynomialDescriptor(self):
        return copy.deepcopy(self.degreePolynomialDescriptor)

    ### Getter for degree of polynomials of target variables.
    def getDegreePolynomialTarget(self):
        return copy.deepcopy(self.degreePolynomialTarget)

    ### Getter for width for Monte Carlo step.
    def getWidthMonteCarlo(self):
        return self.widthMonteCarlo

    ### Setter for width for Monte Carlo step.
    def setWidthMonteCarlo(self, widthMonteCarlo):
        self.widthMonteCarlo = widthMonteCarlo
        return

    ### Getter for number of Monte Carlo samplings.
    def getNSampleMonteCarlo(self):
        return self.nSampleMonteCarlo

    ### Setter for number of Monte Carlo samplings.
    def setNSampleMonteCarlo(self, nSampleMonteCarlo):
        self.nSampleMonteCarlo = nSampleMonteCarlo
        return

    ### Setter for mode of acquisition function.
    def setModeAcquisitionFunction(self, mode):
        self.modeAcquisitionFunction[:] = mode[:]
        self.__setInnerAcquisitionFunction()
        return

    ### Getter for maximal likely precision matrix.
    def getPrecisionMatrixMax(self):
        return copy.deepcopy(self.precisionMatrixMax)

    ### Getter for filename of model parameters.
    def getFilenameModelParameter(self):
        return self.filenameModelParameter

    ### Setter for filename of model parameters.
    def setFilenameModelParameter(self, filename):
        self.filenameModelParameter = filename
        return

    ### Getter for filename of covariance matrix.
    def getFilenameCovarianceMatrix(self):
        return self.filenameCovarianceMatrix

    ### Setter for filename of covariance matrix.
    def setFilenameCovarianceMatrix(self, filename):
        self.filenameCovarianceMatrix = filename
        return

    def getCoef(self):
        return copy.deepcopy(self.coef_)

    ### Getter for computational time for posterior probability.
    def getTimeCalcPostProb(self):
        return self.timeCalcPostProb

    ### Checker of variables.
    def checkVariable(self):
        labelFunction = 'checkVariable'

        flag = True
        if (self.nDescriptor == 0):
            flag = False
        if (self.dimensionDescriptor == 0):
            flag = False
        if (self.degreePolynomialDescriptor is None):
            flag = False
        if (self.nTarget == 0):
            flag = False
        if (self.dimensionTarget == 0):
            flag = False
        if (self.degreePolynomialTarget is None):
            flag = False
        if (self.activeVariable is None):
            flag = False
        if (self.covarianceMatrix is None):
            flag = False
        if (self.precisionMatrixMax is None):
            flag = False
        if (self.nSampleAvailablePattern is None):
            flag = False
        if (self.nAvailablePattern == 0):
            flag = False
        if (self.availablePattern is None):
            flag = False
        return flag
