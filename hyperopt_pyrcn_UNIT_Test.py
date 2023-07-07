####UPDATE: THE UNIT TEST WAS SUCCESSFUL



#This is a unit test to ensure the PYRCN hyperoptimization function is working correctly


from pyRCN_class import pyRCN

reservoir = pyRCN(100) #This automatically calls the hyperoptimization function called by pyRCN_class constructor


print("Printing optimized values\n\n\n", reservoir.esn.get_params)
