#* Author: Febin Sunny (06/05/2023)
#* TED-based thermal crosstalk correction strategy, implemented using theory from:
#* M. Milanizadeh, et al., "Canceling Thermal Cross-Talk Effects in Photonic Integrated Circuits," IEEE JLT, vol. 37, no. 4, 2019

import numpy as np
import sys
import math
from TMatrixGen import *

#* Thermal crosstalk coefficient calculation utilities: BEGIN 
def TMatGen(size:int=10):
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

#* Core algorithm functions: BEGIN 
def Ted_Algorithm(Φ:np.array , δΦ:np.array, T:np.array):
    # Iteratively calculate error function, update phase variables, relaunch TED iteration
    errorVal=[]
    # Velocity term for momentum calculation; not used rn
    velocity =[0]

    iterCount=0

    Φ_old = Φ
    δΦ_old = δΦ

    #* The following term 'Λ_rate' is a learning rate; but becasue of the values involved in this solution space it is set at 1, initially
    Λ_rate = 1

    Λ, P = np.linalg.eig(T)
    P_inv = np.linalg.inv(P)
    T_D = np.diag(Λ)
    # TODO: Possible principle component analysis (PCA) implementation for larger MR array sizes

    #* Sanity check 1
    # For ensuring that the method used to calculate the three variables above make sense 
    # This is direct eigen decomposition [special case of singular value decomposition (SVD) for square matrices]
    # Utilizes eq(2) from Section II of the TED paper
    '''
    T_new = np.linalg.multi_dot([P,T_D,P_inv])
    for l in (T-T_new):
        print(*l)
    '''
    # The error analysis should show that the error between T and T_new are negligible (~16 orders of magnitude below 0)
    
    while(True):
        iterCount+=1
        error=0
        
        Φ_new = TED_iteration(Φ_old , δΦ_old, Λ_rate, P, P_inv, T_D, T)

        δΦ_new = Φ_new-Φ_old
        #* This error term is basically the loss function of this algorithm and 
        #* we shall be trying to minimize it as much as possible, as fast as possible
        error = lossFunctions.meanErrorCalc(δΦ_new)

        Φ_old = Φ_new
        δΦ_old = δΦ_new

        errorVal = np.append(errorVal, error)

        if iterCount>1000:
            sys.exit("Algorithm unable to reach convergence after 1000 iterations. Exiting code.") 
        
        #if math.floor(math.log(np.abs(error),10))<=(-3):
        if convergenceCheck(errorVal):
            #* Threshold based stopping for the algorithm iteration
            print("Last 5 error values leading to convergence",errorVal[-5:])
            return errorVal, δΦ_old
        else:
            #* Learning rate updation based on observed
            #Λ_rate = learningRateMomentumUpdate(errorVal, velocity, Λ_rate)
            Λ_rate=learningRateUpdate(errorVal, Λ_rate)
            if Λ_rate==0:
                return errorVal, δΦ_old
            else:
                continue      

def TED_iteration(Φ:np.array , δΦ:np.array, Λ_rate:float, P:np.array, P_inv:np.array, T_D:np.array, T:np.array):

    δΦ_cap = np.array(np.real(np.dot(T,δΦ)))

    δΨ = np.array(np.real(np.dot(P_inv, δΦ)))
    δΨ_cap = np.array(np.real(np.dot(P_inv, δΦ_cap)))
    
    #* Sanity check 2 
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

    #* Sanity check 3
    # Ensure Φ can be regenerated from the calculated Ψ matrix
    '''
    print(Φ - np.real(np.dot(P,Ψ)))
    '''
    # This check should also generate negligible errors
    
    print(Λ_rate)
    if Λ_rate >0:
        Ψ_new = Ψ + δΨ
    else:
        Ψ_new = Ψ - δΨ
    '''
    print(Λ_rate)
    Ψ_new = Ψ - Λ_rate*δΨ
    '''

    Φ_new = np.array(np.real(np.dot(P,Ψ_new)))
    
    Φ_new = utility.radianManagement(Φ_new)
    
    return Φ_new

def learningRateUpdate(errorVal:np.array, Λ_rate:float):
    if errorVal[-1]>0:
        Λ_rate = np.abs(Λ_rate)*-1
    elif errorVal[-1]<0:
        Λ_rate = np.abs(Λ_rate)
    else:
        Λ_rate=0

    return Λ_rate

# Since the Φ matrix updation process is similar to SGD, we opt for a momentum based learning rate updation
#! This was not the right approach as this modifies the δΨ matrix, artificially bringing down the 
#! number of iterations, but ultimately does not contribute to the phase modification
def learningRateMomentumUpdate(errorVal:np.array, velocity:np.array, Λ_rate:float):
    # Defining momentum coefficient
    β=0.75
    # Momentum term calculation
    v = β*velocity[-1] + (1-β)*errorVal[-1]
    # Learning rate modification
    Λ_rate = Λ_rate*v
    # Updating velocity list
    velocity.append(v)
    # Setting step direction
    if errorVal[-1]>0:
        Λ_rate = np.abs(Λ_rate)*-1
    elif errorVal[-1]<0:
        Λ_rate = np.abs(Λ_rate)
    else:
        Λ_rate=0

    return Λ_rate

def convergenceCheck(errorVal:np.array):
    if len(errorVal)<10:
        return False
    else:
        for i in errorVal[-5:]:
            if math.floor(math.log(np.abs(i),10))>(-3):
                return False
    return True

#* Core algorithm functions: END

def main():
    args = sys.argv[1:]

    if len(args)==0:
        size=10
    elif len(args)==1:
        size= int(args[0])
        
    else:
        print("Size of MR bank expected as input")
        return None
    # generate phase matrices
    Φ , δΦ = utility.phaseChange(size)

    # Generate T-array
    MatObj = TMatrixGen(waveguidePattern='Folded')
    #T = TMatGen(size)

    T = MatObj.randomTMatrixGenerator()

    iterError, ΔΦ =Ted_Algorithm(Φ , δΦ, T)

    utility.plotGraph(iterError)

    print("Original phase values:", Φ)
    print("Final phase values:", utility.radianManagement(Φ+ΔΦ))

    return None

if __name__ == '__main__':
    main()