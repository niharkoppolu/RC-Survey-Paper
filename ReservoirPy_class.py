from Framework_Superclass import Reservoir_Framework
from reservoirpy.observables import mse, nrmse, rmse
import matplotlib.pyplot as plt

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 1: ReservoirPy (import of framework is placed in init)
class ReservoirPy(Reservoir_Framework): #superclass containing in brackets

    def __init__(self, Set_Num_Nodes):
        from reservoirpy.nodes import Reservoir, Ridge
        super().__init__(Set_Num_Nodes) #must call init function of superclass
        

        self.reservoir = Reservoir(self.reservoir_nodes, lr = self.leakage_rate, sr = self.spectral_radius)
        self.readout = Ridge(ridge=1e-7)

        self.esn_model = self.reservoir >> self.readout #connects reservoir and ridge

        pass

    #override abstract Train method
    def Train(self):
        
        self.X_train = self.mackey_glass_final[0:self.training_timesteps] #must add self for class variables
        self.Y_train = self.mackey_glass_final[1:self.training_timesteps + 1] 

        self.esn_model = self.esn_model.fit(self.X_train, self.Y_train, warmup=10) #training RC
        #print(self.reservoir.is_initialized, self.readout.is_initialized, self.readout.fitted) #used to check if training done

        pass

    
    #override abstract Test method
    def Test(self):

        Y_pred = self.esn_model.run(self.mackey_glass_final[self.test_set_begin : self.test_set_end])
        Y_actual = self.mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1]

        #used to graph
        """
        plt.figure(figsize=(10, 3))
        plt.title("Predicted and Actual Mackey_Glass Timeseries.")
        plt.xlabel("$t$")
        plt.plot(Y_pred, label="Predicted ", color="blue")
        plt.plot(Y_actual, label="Real ", color="red")
        plt.legend()
        plt.show()
        """
        

        #this is used to calculate nrmse and mse
        self.calculated_nrmse = nrmse(Y_pred, Y_actual)
        self.calculated_mse = (Y_pred, Y_actual)

        return self.calculated_nrmse


    #override abstract Reset method
    def Reset_Reservoir(self):
        from reservoirpy.nodes import Ridge

        self.readout = Ridge(ridge=1e-7)
        self.esn_model = self.reservoir >> self.readout #connects reservoir and ridge

        pass

