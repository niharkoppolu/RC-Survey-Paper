#This is a unit test to ensure the reservoirPy hyperoptimization function is working correctly


from ReservoirPy_class import ReservoirPy

reservoir = ReservoirPy(100) #This automatically calls the hyperoptimization function called by pyRCN_class constructor


print("Printing reservoirpy optimized values\n\n\n", reservoir.best_params)