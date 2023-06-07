# Author: Febin Sunny (06/05/2023)
# TED-based thermal crosstalk correction strategy, implemented from:
# M. Milanizadeh, et al., "Canceling Thermal Cross-Talk Effects in Photonic Integrated Circuits," IEEE JLT, vol. 37, no. 4, 2019

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import sys
from pprint import pprint
import math
from pandas import *


def phaseChange(size:int=10):
    # Generate phae matrices and calculate difference to kick off the evaluation
    # Considering phase values to be in radians ranging from 0 to 2π
    stopLim=int(round(2*np.pi,5)*pow(10,5))

    # Original phase variables Φ
    Φ_original = np.random.randint(0,stopLim,(size))/pow(10,5)
    # New phase values to be implemented in the MR bank
    Φ =  np.random.randint(0,stopLim,(size))/pow(10,5)
    # δΦ or change in phase value, which is the progenitor of thermal error 
    δΦ = Φ_original-Φ
    
    return Φ , δΦ

def radianManagement(inputPhaseMat:np.array):
    # This utility function ensures the Φ matrices generated stay within the 0 to 2π range
    # Phase values generated from the calculations can overflow and underflow this range
    # To make sense of the iterations the Φ matrices should be processed to retain the range 
    phaseMat=[]

    twoπ=int(round(2*np.pi,5)*pow(10,5))

    for i in inputPhaseMat:
        #print(i)
        if i>0:
            phaseMat=np.append(phaseMat, (i*pow(10,5)%twoπ)/pow(10,5))
        elif i<0:
            temp=i*pow(10,5)%twoπ
            phaseMat=np.append(phaseMat, (twoπ-temp)/pow(10,5))
        else:
            phaseMat=np.append(phaseMat, 0)
    return np.array(phaseMat)

def TMatrixGen(size:int=10):
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

def TMatrixGen_rand(size:int=10):
    T = np.random.rand(size,size)
    np.fill_diagonal(T, 1)

    return T

def Ted_Algorithm(Φ:np.array , δΦ:np.array, T:np.array, Num_iter:int=20):
    # Iteratively calculate error function, update phase variables, relaunch TED iteration
    errorVal=[]

    Φ_old = Φ
    δΦ_old = δΦ

    stepDir='+'

    Λ, P = np.linalg.eig(T)
    P_inv = np.linalg.inv(P)
    T_D = np.diag(Λ)

    for i in range(Num_iter):
        error=0
        Φ_new = TED_iteration(Φ_old , δΦ_old, stepDir, P, P_inv, T_D, T)

        δΦ_new = Φ_new-Φ_old
        #print(δΦ_new)
        for i in δΦ_new:
            error=error+i
        error=error/(len(δΦ_new))

        Φ_old = Φ_new
        δΦ_old = δΦ_new

        errorVal = np.append(errorVal, error)
        if math.ceil(math.log(np.abs(error),10))<=(-3):
            return errorVal, δΦ_old
        else:
            if error>0:
                stepDir='-'
            elif error<0:
                stepDir='+'
            else:
                return errorVal, δΦ_old

    return errorVal, δΦ_old

def TED_iteration(Φ:np.array , δΦ:np.array, stepDir:str, P:np.array, P_inv:np.array, T_D:np.array, T:np.array):

    δΦ_cap = np.array(np.real(np.dot(T,δΦ)))

    # Sanity check 1
    # For ensuring that the method used to calculate the three variables above make sense 
    # This is direct eigen decomposition [special case of singular value decomposition (SVD) for square matrices]
    # Utilizes eq(2) from Section II of the TED paper
    '''
    T_new = np.linalg.multi_dot([P,T_D,P_inv])
    for l in (T-T_new):
        print(*l)
    '''
    # The error analysis should show that the error between T and T_new are negligible (~16 orders of magnitude below 0)

    δΨ = np.array(np.real(np.dot(P_inv, δΦ)))
    δΨ_cap = np.array(np.real(np.dot(P_inv, δΦ_cap)))
    
    # Sanity check 2 
    # To ensure δΨ_cap calculation is on track
    # From Section II, eq(2) and eq(3) of the TED paper:
    # δΨ_cap=T_D.δΨ
    # While simultaneously:
    # δΨ_cap=P_inv.δΦ_cap
    '''
    print(δΨ_cap-np.real(np.dot(T_D, δΨ)))
    '''
    # The error analysis should show that the error between both methods of calculations are negligible (~16 orders of magnitude below 0)
    # This means that the relevant matrices generated are correct, and our implementation methodology so far tracks

    Ψ = np.array(np.real(np.dot(P_inv, Φ)))

    # Sanity check 3
    # Ensure Φ can be regenerated from the calculated Ψ matrix
    '''
    print(Φ - np.real(np.dot(P,Ψ)))
    '''
    # This check should also generate negligible errors
    if stepDir=='+':
        Ψ_new = Ψ + δΨ
    elif stepDir=='-':
        Ψ_new = Ψ - δΨ
    #print("New Ψ matrix:",Ψ_new)
    Φ_new = np.array(np.real(np.dot(P,Ψ_new)))
    
    Φ_new = radianManagement(Φ_new)
    
    return Φ_new

def plotGraph(errorList:np.array):
    print("Final error list:",errorList)

    plt.plot(np.arange(0,len(errorList),1),errorList)
    plt.show()

    return None

def main():
    args = sys.argv[1:]

    if len(args)==0:
        size=10
        Num_iter=20
    elif len(args)==2:
        size= int(args[0])
        Num_iter=int(args[1])
        
    else:
        print("Size of MR bank and number of iterations expected as inputs")
        return None
    # generate phase matrices
    Φ , δΦ = phaseChange(size)

    # Generate T-array 
    T = TMatrixGen(size)

    #T = TMatrixGen_rand(size)

    iterError, ΔΦ =Ted_Algorithm(Φ , δΦ, T, Num_iter)

    plotGraph(iterError)

    print("Original phase values:", Φ)
    print("Final phase values:", (Φ+ΔΦ))

    return None

if __name__ == '__main__':
    main()