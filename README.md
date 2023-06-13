# Thermal Eigenmode Decomposition
Thermal Eigenmode Decomposition (TED) algorithm implementation for thermal crosstalk mitigation in photonic systems
TED-based thermal crosstalk correction strategy, implemented using theory from: M. Milanizadeh, et al., "Canceling Thermal Cross-Talk Effects in Photonic Integrated Circuits," IEEE JLT, vol. 37, no. 4, 2019

This implementartion is focused on microring resonator banks and how thermal crosstalk can be corrected in them.

The algorithm performs an iterative correction of phase input to the microring resonator (MR) banks, so that through eigenmode decomposition (singualr value decomposition for square matrices) we can eliminate 
thermal crosstalk or for the sake of system level design and reduction of iterations, reduce it below a threshold.

For the sake of scalability of the algorithm for varying MR bank sizes, I have used multivariate polynomial regression for curve fitting over thermal crosstalk data obtained through HEAT simulations. 
For clarification: Lumerical HEAT is a multiphysics simulator and can simulate variations in light-matter interaction as the environment temperature changes.

Currently the tool can evalaute thermal crosstalk over several MR arranegements, listed as follows:
### Straight arrangement:
Waveguide geometry is one straight waveguide with MRs placed at regular intervals; like so:

            # O     O       O       O       O       O       O     O       O       O 
            #-------------------------------------------------------------------------
### Folded arrangement:            
Waveguide geometry is a folded waveguide with MRs placed at regular intervals; like so:

            # O     O       O       O       O
            #-----------------------------------
            #                                   |
            #-----------------------------------
            # O     O       O       O       O
### Staggered arrangement:
Waveguide geometry is a straight waveguide with MRs placed at regular staggered intervals; like so (note the numbering):

            # O1     O3       O5       O7       O9
            #---------------------------------------
            #    O2      O4       O6       O8      
            
### Opposite arrangement:
Waveguide geometry is a straight waveguide with MRs placed at regular intervals, opposite to each other; like so (note the numbering):

            # O1     O3       O5       O7       O9
            #--------------------------------------
            # O2     O4       O6       O8      O10
            
### Staggered-fold arrangement:
Waveguide geometry is a straight waveguide with MRs placed at regular staggered intervals; like so (note the numbering):

            # O1     O2       O3       O4       O5
            #---------------------------------------
            #    O9      O8       O7       O6        
            
### Opposite-fold arrangement:
Waveguide geometry is a straight waveguide with MRs placed at regular intervals, opposite to each other; like so (note the numbering):

            # O1     O2       O3       O4       O5
            #--------------------------------------
            # O10    O9       O8       O7      O6
 
 ## TED Configuration
I have provided a simple YAML based configurtion file which may be expanded in the future to provide more functionalities to the tool.
