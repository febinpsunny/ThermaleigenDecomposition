#* Author: Febin Sunny (06/09/2023)
#* Various utility classes including the scalability enabling TMatrixGen class

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.stats import mode
import warnings
import functools
import sys
import math

warnings.filterwarnings("always", category=DeprecationWarning)

class utility:
    # Simple printing utility to show matrices better
    @staticmethod
    def matrixPrint(matrix:np.array):
        for i in matrix:
            print(*i)
        return None
    
    # Graphing utility
    @staticmethod
    def plotGraph(errorList:np.array):
        plt.plot(np.arange(0,len(errorList),1),errorList)
        plt.show()
        return None
    
    @staticmethod
    def phaseChange(size:int=10):
        # Generate phae matrices and calculate difference to kick off the algorithm
        # Considering phase values to be in radians ranging from 0 to 2π
        stopLim=int(round(2*np.pi,5)*pow(10,5))

        # Original phase variables Φ
        Φ_original = np.random.randint(0,stopLim,(size))/pow(10,5)
        # New phase values to be implemented in the MR bank
        Φ =  np.random.randint(0,stopLim,(size))/pow(10,5)
        # δΦ or change in phase value, which is the progenitor of thermal error 
        δΦ = Φ_original-Φ
        
        return Φ , δΦ

    @staticmethod
    def radianManagement(inputPhaseMat:np.array):
        # This utility function ensures the Φ matrices generated stay within the 0 to 2π range
        # Phase values generated from the calculations can overflow and underflow this range
        # To make sense of the iterations the Φ matrices should be processed to retain the range 
        phaseMat=[]

        twoπ=int(round(2*np.pi,5)*pow(10,5))

        for i in inputPhaseMat:
            if i>0:
                phaseMat=np.append(phaseMat, (i*pow(10,5)%twoπ)/pow(10,5))
            elif i<0:
                temp=i*pow(10,5)%twoπ
                phaseMat=np.append(phaseMat, (twoπ-temp)/pow(10,5))
            else:
                phaseMat=np.append(phaseMat, 0)
        return np.array(phaseMat)
    
    @staticmethod
    def deprecated(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn("This function is deprecated. Use TMatrixGenerator() instead", category=DeprecationWarning)
            return func(*args, **kwargs)
        return wrapper

class lossFunctions:
    #* Loss function calculation utilities: BEGIN
    @staticmethod 
    def meanErrorCalc(Φ:np.array):
        error =0
        for i in Φ:
            error=error+i
            error=error/(len(Φ))
        
        return error
    @staticmethod
    def medianErrorCalc(Φ:np.array):
        return np.median(Φ)

    @staticmethod
    def modeErrorCalc(Φ:np.array):
        # Only concerning this iteration with primary mode
        return mode(Φ)[0]

    @staticmethod
    def maxErrorCalc(Φ:np.array):
        # This needs careful consideration as we need maximum absolute value, but should preserve 
        # the sign of the value; i.e. np.max() alone is not useful and will lead to a constant 
        # +ve step direction leading to the algorithm never converging
        max=0
        min=0
        for i in Φ:
            if i> max:
                max =i
            if i<min:
                min = i
        if np.abs(max)>np.abs(min):
            return max
        else:
            return min

    # Modified MSE which preserves the directionality of the error
    @staticmethod
    def meanSquareErrorMod(Φ:np.array):
        squaresum=0
        for i in Φ:
            squaresum= squaresum+(i*i)
        MSE = squaresum/len(Φ)

        return MSE*Φ[-1]/np.abs(Φ[-1])

    #* Loss function calculation utilities: END


class TMatrixGen():

    def __init__(self, arraySizeMR:np.int32=10, radiusMR:np.float32=5000, gapMRWG:np.float32=5, interMRdistance:np.float32=700, widthWG:np.float32=450, 
                 interWGdistance:np.float32=2000, waveguidePattern:str='Straight', polydegree:np.int32=5, Φ_input:np.array=[], distance:np.array=[],
                 δΦ:np.array=[]):
        self.poly = None
        self.model = None
        self.countMR = arraySizeMR
        self.radiusMR = radiusMR
        self.gapMRWG = gapMRWG
        self.interMRdistance = interMRdistance
        self.widthWG = widthWG
        self.interWGdistance = interWGdistance
        self.waveguidePattern = waveguidePattern
        self.polydegree = polydegree
        self.Φ_input = Φ_input
        self.distance = distance
        self.δΦ = δΦ

    def polyFitGenerator(self):
        # TODO: Accept a csv file or similar config file with data to perform Polynomial regression

        # Controls row-wise values of the δΦ; data amputation performed to match dimensions with distance[]
        Φ_input = np.array([0, 0.048794847, 0.194480236, 0.767453116, 1.187525765, 1.690515667, 2.271208328, 3.644105958])

        # Controls column-wise values of the δΦ
        distance = np.array([700, 5132, 11400, 13926, 22100, 23836, 32800, 34117])

        # Dependant values; data amputation performed to match dimensions with distance[]
        δΦ =np.array([[0,	        0,	            0,	            0,	            0,	            0,	            0,	            0],
                    [0.011050455,	0.002019767,	0.000281602,	0.000145656,	1.94208e-05,	1.94208e-05,	0,	            0],
                    [0.044065873,	0.008088778,	0.001136119,	0.000572915,	0.000106815,	8.73938e-05,	3.88417e-05,	1.94208e-05],
                    [0.174301989,	0.032102639,	0.004466792,	0.002281948,	0.000407838,	0.000330154,	9.71042e-05,	9.71042e-05],
                    [0.27021178,	0.049882413,	0.006952659,	0.003544302,	0.000650598,	0.000514652,	0.000155367,	0.000155367],
                    [0.385474433,	0.071342435,	0.009943467,	0.005078548,	0.00092249,	    0.000737992,	0.000252471,	0.000203919],
                    [0.519080064,	0.096405022,	0.013429507,	0.006865265,	0.001242933,	0.001000173,	0.000310733,	0.000281602],
                    [0.83714478,	0.15662903,	    0.021848439,	0.011157269,	0.002019767,	0.00163135,	    0.000524363,	0.00045639]])
        
        # Generate coordinate pairs for the available independant variable data
        X = np.column_stack((np.repeat(Φ_input, len(distance)), np.tile(distance, len(Φ_input))))

        poly = PolynomialFeatures(degree=self.polydegree)
        X_poly = poly.fit_transform(X)

        δΦ = δΦ.flatten()

        model = LinearRegression()
        model.fit(X_poly, δΦ)
        
        self.poly=poly
        self.model = model

        return None

    def waveguideGeometryManger(self):
        #* Ensure radiusMR, interMRdistance, and interWGdistance are provided in nanometers (nm)
        # Generate a list of distances dependant on the values provided
            
        if self.waveguidePattern=='Straight':
            # Waveguide geometry is one straight waveguide with MRs placed at regular intervals; like so:
            # O     O       O       O       O       O       O     O       O       O 
            #-------------------------------------------------------------------------
            distanceMatrix=[]
            for i in range(self.countMR):
                distance_row=[]
                for j in range(self.countMR):
                    distance = np.abs(j-i)*(2*self.radiusMR+self.interMRdistance)
                    distance_row.append(distance)
                distanceMatrix.append(distance_row)
        elif self.waveguidePattern=='Folded':
            # Waveguide geometry is a folded waveguide with MRs placed at regular intervals; like so:
            # O     O       O       O       O
            #-----------------------------------
            #                                   |
            #-----------------------------------
            # O     O       O       O       O
            distanceMatrix=[]
            fold = np.int32(np.ceil(self.countMR/2))
            for i in range(self.countMR):
                if i<fold:
                    foldflagMr1=0
                else:
                    foldflagMr1=1
                distance_row=[]
                for j in range(self.countMR):
                    if j<fold:
                        foldflagMr2=0
                    else:
                        foldflagMr2=1
                    if foldflagMr2==foldflagMr1:
                        # If the MRs are on the same side of the fold utilize regular distance calculation
                        distance = np.abs(j-i)*(2*self.radiusMR + self.interMRdistance)
                    else:
                        # If the MRs are on the opposite sides of the fold utilize Pythagorian theorem for distance calculation
                        vertical = 2*self.radiusMR + self.interWGdistance + 2*self.widthWG + 2*self.gapMRWG
                        horizontal = np.abs(((self.countMR-1)-i)-j)*(2*self.radiusMR + self.interMRdistance)
                        distance = np.sqrt((vertical**2)+(horizontal**2))
                    distance_row.append(distance)
                distanceMatrix.append(distance_row)
        elif self.waveguidePattern=='Staggered':
            # Waveguide geometry is a straight waveguide with MRs placed at regular staggered intervals; like so (note the numbering):
            # O1     O3       O5       O7       O9
            #---------------------------------------
            #    O2      O4       O6       O8          
            distanceMatrix=[]
            for i in range(self.countMR):
                if i%2==0:
                    foldflagMr1=0
                else:
                    foldflagMr1=1
                distance_row=[]
                for j in range(self.countMR):
                    if j%2==0:
                        foldflagMr2=0
                    else:
                        foldflagMr2=1
                    if foldflagMr2==foldflagMr1:
                        # If the MRs are on the same side of the fold utilize regular distance calculation
                        distance = np.abs(j-i)*(4*self.radiusMR + self.interMRdistance)
                    else:
                        # If the MRs are on the opposite sides of the fold utilize Pythagorian theorem for distance calculation
                        vertical = 2*self.radiusMR + self.widthWG + 2*self.gapMRWG
                        horizontal = np.abs(((self.countMR-1)-i)-j)*(2*self.radiusMR + 0.5*self.interMRdistance)
                        distance = np.sqrt((vertical**2)+(horizontal**2))
                    distance_row.append(distance)
                distanceMatrix.append(distance_row)               
        elif self.waveguidePattern=='Opposite':
            # Waveguide geometry is a straight waveguide with MRs placed at regular intervals, opposite to each other; like so (note the numbering):
            # O1     O3       O5       O7       O9
            #--------------------------------------
            # O2     O4       O6       O8      O10
            distanceMatrix=[]
            for i in range(self.countMR):
                if i%2==0:
                    foldflagMr1=0
                else:
                    foldflagMr1=1
                distance_row=[]
                for j in range(self.countMR):
                    if j%2==0:
                        foldflagMr2=0
                    else:
                        foldflagMr2=1
                    if foldflagMr2==foldflagMr1:
                        # If the MRs are on the same side of the fold utilize regular distance calculation
                        distance = np.abs(j-i)*(2*self.radiusMR + self.interMRdistance)
                    else:
                        # If the MRs are on the opposite sides of the fold utilize Pythagorian theorem for distance calculation
                        vertical = 2*self.radiusMR + self.widthWG + 2*self.gapMRWG
                        horizontal = np.abs(((self.countMR-1)-i)-j)*(2*self.radiusMR + self.interMRdistance)
                        distance = np.sqrt((vertical**2)+(horizontal**2))
                    distance_row.append(distance)
                distanceMatrix.append(distance_row)
        elif self.waveguidePattern=='Staggered-fold':
            # Waveguide geometry is a straight waveguide with MRs placed at regular staggered intervals; like so (note the numbering):
            # O1     O2       O3       O4       O5
            #---------------------------------------
            #    O9      O8       O7       O6          
            distanceMatrix=[]
            fold = np.int32(np.ceil(self.countMR/2))
            for i in range(self.countMR):
                if i<fold:
                    foldflagMr1=0
                else:
                    foldflagMr1=1
                distance_row=[]
                for j in range(self.countMR):
                    if j<fold:
                        foldflagMr2=0
                    else:
                        foldflagMr2=1
                    if foldflagMr2==foldflagMr1:
                        # If the MRs are on the same side of the fold utilize regular distance calculation
                        distance = np.abs(j-i)*(4*self.radiusMR + self.interMRdistance)
                    else:
                        # If the MRs are on the opposite sides of the fold utilize Pythagorian theorem for distance calculation
                        vertical = 2*self.radiusMR + self.widthWG + 2*self.gapMRWG
                        horizontal = np.abs(((self.countMR-1)-i)-j)*(2*self.radiusMR + 0.5*self.interMRdistance)
                        distance = np.sqrt((vertical**2)+(horizontal**2))
                    distance_row.append(distance)
                distanceMatrix.append(distance_row)               
        elif self.waveguidePattern=='Opposite-fold':
            # Waveguide geometry is a straight waveguide with MRs placed at regular intervals, opposite to each other; like so (note the numbering):
            # O1     O2       O3       O4       O5
            #--------------------------------------
            # O10    O9       O8       O7      O6
            distanceMatrix=[]
            fold = np.int32(np.ceil(self.countMR/2))
            for i in range(self.countMR):
                if i<fold:
                    foldflagMr1=0
                else:
                    foldflagMr1=1
                distance_row=[]
                for j in range(self.countMR):
                    if j<fold:
                        foldflagMr2=0
                    else:
                        foldflagMr2=1
                    if foldflagMr2==foldflagMr1:
                        # If the MRs are on the same side of the fold utilize regular distance calculation
                        distance = np.abs(j-i)*(2*self.radiusMR + self.interMRdistance)
                    else:
                        # If the MRs are on the opposite sides of the fold utilize Pythagorian theorem for distance calculation
                        vertical = 2*self.radiusMR + self.widthWG + 2*self.gapMRWG
                        horizontal = np.abs(((self.countMR-1)-i)-j)*(2*self.radiusMR + self.interMRdistance)
                        distance = np.sqrt((vertical**2)+(horizontal**2))
                    distance_row.append(distance)
                distanceMatrix.append(distance_row)

        return np.array(distanceMatrix)
    
    #* Thermal crosstalk coefficient calculation utilities: BEGIN
    def thermalCrosstalkEstimator(self, distance:np.float32, Φ_value:np.float32=1):
        X = self.poly.transform([[Φ_value, distance]])
        thermalCrosstalkCoefficient = self.model.predict(X)
        return thermalCrosstalkCoefficient

    def TMatrixGenerator(self, distanceMatrix:np.array):
        self.polyFitGenerator()
        T=[]
        for row in distanceMatrix:
            T_row=[]
            for i in row:
                T_row.append(self.thermalCrosstalkEstimator(distance=i))
            T.append(np.array(T_row).flatten())
        return T
    
    @utility.deprecated
    def TMatGen(self):
        size = self.countMR
        # TODO: This assertion is just a place holder; with proper T matrix generation through extrapolation this limiter can be lifted
        assert size<=10, "Current T matrix generation only supports up to 10 MRs."
        
        # This is an accurate matrix implementation, obtained from HEAT simulations using a "folded" 10 MR MR-bank
        # TODO: This function content needs to be replaced by the extrapolation function from these simulations for scalability!! 
        T =   np.array([[1, 0.2270, 0.0072, 0.0004, 0.0001, 0.0009, 0.0020, 0.0088, 0.0431, 0.2270],    #0
                        [0.2270, 1, 0.2270, 0.0072, 0.0004, 0.0020, 0.0088, 0.0431, 0.2270, 0.0431],    #1
                        [0.0072, 0.2270, 1, 0.2270, 0.0072, 0.0088, 0.0431, 0.2270, 0.0431, 0.0088],    #2
                        [0.0004, 0.0072, 0.2270, 1, 0.2270, 0.0431, 0.2270, 0.0431, 0.0088, 0.0020],    #3
                        [0.0001, 0.0004, 0.0072, 0.2270, 1, 0.2270, 0.0431, 0.0088, 0.0020, 0.0009],    #4
                        [0.0009, 0.0020, 0.0088, 0.0431, 0.2270, 1, 0.2270, 0.0072, 0.0004, 0.0001],    #5
                        [0.0020, 0.0088, 0.0431, 0.2270, 0.0431, 0.2270, 1, 0.2270, 0.0072, 0.0004],    #6
                        [0.0088, 0.0431, 0.2270, 0.0431, 0.0088, 0.0072, 0.2270, 1, 0.2270, 0.0072],    #7
                        [0.0431, 0.2270, 0.0431, 0.0088, 0.0020, 0.0004, 0.0072, 0.2270, 1, 0.2270],    #8
                        [0.2270, 0.0431, 0.0088, 0.0020, 0.0009, 0.0001, 0.0004, 0.0072, 0.2270, 1]])   #9
        
        T_new = np.array(T[0:size,0:size])

        return T_new


    #* Thermal crosstalk coefficient calculation utilities: END 

    def randomTMatrixGenerator(self):
        T = np.random.rand(self.countMR, self.countMR)
        np.fill_diagonal(T, 1)
        return T

#? Testing Area
# MatObj = TMatrixGen(waveguidePattern='Staggered')
# distanceMatrix = MatObj.waveguideGeometryManger()
# utility.matrixPrint(distanceMatrix)
# utility.matrixPrint(MatObj.TMatrixGenerator(distanceMatrix))