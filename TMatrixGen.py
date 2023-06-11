import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sys
import math

def polyFitGenerator(polydegree:np.int16=2):
    # TODO: Accept a csv file or similar config file with data to perform Polynomial regression

    # Controls row-wise values of the δΦ; data amputation performed to match dimensions with distance[]
    Φ_input = [0, 0.048794847, 0.194480236, 0.767453116, 1.187525765, 1.690515667, 2.271208328, 3.644105958]

    # Controls column-wise values of the δΦ
    distance = [700, 5132, 11400, 13926, 22100, 23836, 32800, 34117]

    # Independant values; data amputation performed to match dimensions with distance[]
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

    poly = PolynomialFeatures(degree=polydegree)
    X_poly = poly.fit_transform(X)

    δΦ = δΦ.flatten()

    model = LinearRegression()
    model.fit(X_poly, δΦ)

    return poly, model

def waveguideGeometryManger(countMR:np.int32, radiusMR:np.float32=5000, interMRdistance:np.float32=700, interWGdistance:np.float32=2000, waveguidePattern:str='Straight'):
    #* Ensure radiusMR, interMRdistance, and interWGdistance are provided in nanometers (nm)
    # Generate a list of distances dependant on the values provided
        
    if waveguidePattern=='Straight':
        # Waveguide geometry is one straight waveguide with MRs placed at regular intervals; like so:
        # O     O       O       O       O       O
        #----------------------------------------------
        distanceMatrix=[]
        for i in range(countMR):
            distance_row=[]
            for j in range(countMR):
                distance = np.abs(j-i)*(radiusMR+interMRdistance)
                distance_row.append(distance)
            distanceMatrix.append(distance_row)
    elif waveguidePattern=='Folded':
        # Waveguide geometry is a folded waveguide with MRs placed at regular intervals; like so:
        # O     O       O       O       O
        #-----------------------------------
        #                                   |
        #-----------------------------------
        # O     O       O       O       O
        distanceMatrix=[]
        fold = np.int32(np.ceil(countMR/2))
        for i in range(countMR):
            if i<fold:
                foldflagMr1=0
            else:
                foldflagMr1=1
            distance_row=[]
            for j in range(countMR):
                if j<fold:
                    foldflagMr2=0
                else:
                    foldflagMr2=1
                if foldflagMr2==foldflagMr1:
                    # If the MRs are on the same side of the fold utilize regular distance calculation
                    distance = np.abs(j-i)*(radiusMR+interMRdistance)
                else:
                    # If the MRs are on the sopposite sides of the fold utilize Pythagorian theorem for distance calculation
                    vertical = radiusMR+interWGdistance
                    horizontal = np.abs(((countMR-1)-i)-j)*(radiusMR+interMRdistance)
                    distance = np.sqrt((vertical**2)+(horizontal**2))
                distance_row.append(distance)
            distanceMatrix.append(distance_row)
                                    
    return distanceMatrix

def thermalCrosstalkEstimator(poly, model, ):
    return None

def TMatrixGenerator():
    T=[]
    return T

def randomTMatrixGenerator(size:int=10):
    T = np.random.rand(size,size)
    np.fill_diagonal(T, 1)

    return T


def matrixPrint(matrix:np.array):
    for i in matrix:
        print(*i)
    return None

matrixPrint(waveguideGeometryManger(10))#, 5000, 700, 2000,'Folded'))