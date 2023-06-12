#* Author: Febin Sunny (06/05/2023)
#* TED-based thermal crosstalk correction strategy, implemented using theory from:
#* M. Milanizadeh, et al., "Canceling Thermal Cross-Talk Effects in Photonic Integrated Circuits," IEEE JLT, vol. 37, no. 4, 2019

import numpy as np
import sys
import math
from TMatrixGen import *
from Configparser import *

class ThermalEigenmodeDecomposition:

    def __init__(self, Φ:np.array , δΦ:np.array, T:np.array, Λ_rate:np.float32=1):
        self.T = T
        self.P = None
        self.P_inv = None
        self.T_D = None
        self.Λ = None
        self.Φ = Φ
        self.δΦ = δΦ
        self.Λ_rate = Λ_rate
        

    #* Core algorithm functions: BEGIN 
    def TED_Algorithm(self):
        # Iteratively calculate error function, update phase variables, relaunch TED iteration
        errorVal=[]
        # Velocity term for momentum calculation; not used rn
        velocity =[0]

        iterCount=0

        Φ_old = self.Φ
        δΦ_old = self.δΦ

        #* The following term 'Λ_rate' is a learning rate; but becasue of the values involved in this solution space it is set at 1, initially
        Λ_rate = self.Λ_rate

        self.Λ, self.P = np.linalg.eig(self.T)
        self.P_inv = np.linalg.inv(self.P)
        self.T_D = np.diag(self.Λ)
        # TODO: Possible principle component analysis (PCA) implementation for larger MR array sizes

        #* Sanity check 1
        # For ensuring that the method used to calculate the three variables above make sense 
        # This is direct eigen decomposition [special case of singular value decomposition (SVD) for square matrices]
        # Utilizes eq(2) from Section II of the TED paper
        '''
        T_new = np.linalg.multi_dot([self.P,self.T_D,self.P_inv])
        for l in (self.T-T_new):
            print(*l)
        '''
        # The error analysis should show that the error between T and T_new are negligible (~16 orders of magnitude below 0)
        
        while(True):
            iterCount+=1
            error=0
            
            Φ_new = self.TED_iteration(Φ_old , δΦ_old, Λ_rate)

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
            if self.convergenceCheck(errorVal):
                #* Threshold based stopping for the algorithm iteration
                print("Last 5 error values leading to convergence",errorVal[-5:])
                return errorVal, δΦ_old
            else:
                #* Learning rate updation based on observed
                #Λ_rate = learningRateMomentumUpdate(errorVal, velocity, Λ_rate)
                Λ_rate=self.learningRateUpdate(errorVal, Λ_rate)
                if Λ_rate==0:
                    return errorVal, δΦ_old
                else:
                    continue      

    def TED_iteration(self, Φ:np.array , δΦ:np.array, Λ_rate:float):

        δΦ_cap = np.array(np.real(np.dot(self.T,δΦ)))

        δΨ = np.array(np.real(np.dot(self.P_inv, δΦ)))
        δΨ_cap = np.array(np.real(np.dot(self.P_inv, δΦ_cap)))
        
        #* Sanity check 2 
        # To ensure δΨ_cap calculation is on track
        # From Section II, eq(2) and eq(3) of the TED paper:
        # δΨ_cap=T_D.δΨ
        # While simultaneously:
        # δΨ_cap=P_inv.δΦ_cap
        '''
        print(δΨ_cap-np.real(np.dot(self.T_D, δΨ)))
        '''
        # The error analysis should show that the error between both methods of calculations are negligible (~16 orders of magnitude below 0)
        # This means that the relevant matrices generated are correct, and our implementation methodology so far tracks

        Ψ = np.array(np.real(np.dot(self.P_inv, Φ)))

        #* Sanity check 3
        # Ensure Φ can be regenerated from the calculated Ψ matrix
        '''
        print(Φ - np.real(np.dot(self.P,Ψ)))
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

        Φ_new = np.array(np.real(np.dot(self.P,Ψ_new)))
        
        Φ_new = utility.radianManagement(Φ_new)
        
        return Φ_new

    def learningRateUpdate(self, errorVal:np.array, Λ_rate:float):
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
    def learningRateMomentumUpdate(self, errorVal:np.array, velocity:np.array, Λ_rate:float):
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

    def convergenceCheck(self, errorVal:np.array):
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
        argList, Φ_input, distance, δΦ_thermal = yamlConfigParser()
    elif len(args)==1:
        argList, Φ_input, distance, δΦ_thermal = yamlConfigParser(args[0])        
    else:
        print("Config file expected as commandline input")
        return None
    
    print(argList)

    # generate phase matrices
    Φ , δΦ = utility.phaseChange(argList[0])

    # Generate T-array
    MatObj = TMatrixGen(*argList, Φ_input, distance, δΦ_thermal)
    #T = TMatGen(size)

    T = MatObj.TMatGen()

    # distanceMatrix = MatObj.waveguideGeometryManger()
    # T = MatObj.TMatrixGenerator(distanceMatrix)

    Algorithm = ThermalEigenmodeDecomposition(Φ , δΦ, T, 1)

    iterError, ΔΦ =Algorithm.TED_Algorithm()

    utility.plotGraph(iterError)

    print("Original phase values:", Φ)
    print("Final phase values:", utility.radianManagement(Φ+ΔΦ))

    return None

if __name__ == '__main__':
    main()