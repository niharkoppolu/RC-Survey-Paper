from Framework_Superclass import Reservoir_Framework
from reservoirpy.observables import mse, nrmse, rmse
import matplotlib.pyplot as plt

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 3: pyRCN (import of framework is placed in init)
class pyRCN(Reservoir_Framework): #superclass containing in brackets

    def __init__(self, Set_Num_Nodes):
        from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
        from sklearn.linear_model import Ridge

        super().__init__(Set_Num_Nodes) #must call init function of superclass

        # Hyperparameter optimization ESN
        initially_fixed_params = {'hidden_layer_size': self.reservoir_nodes,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': self.leakage_rate,
                          'bidirectional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'requires_sequence': False,
                          'spectral_radius': self.spectral_radius}

        self.esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params) 
        #old self.esn = ESNRegressor()
        self.esn
        pass

    #override abstract Train method
    def Train(self):
        
        self.X_train = self.mackey_glass_final[0:self.training_timesteps] #must add self for class variables
        self.Y_train = self.mackey_glass_final[1:self.training_timesteps + 1] 

        self.esn.fit(self.X_train.reshape(-1, 1), self.Y_train)
        pass

    
    #override abstract Test method
    def Test(self):

        Y_pred = self.esn.predict(self.mackey_glass_final[self.test_set_begin : self.test_set_end])
        Y_actual = self.mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1]
        """
        #used to graph
        
        plt.figure(figsize=(10, 3))
        plt.title("Predicted and Actual Mackey_Glass Timeseries.")
        plt.xlabel("$t$")
        plt.plot(Y_pred, label="Predicted ", color="blue")
        #plt.plot(Y_actual, label="Real ", color="red")
        plt.legend()
        plt.show()
        """
        
        

        #this is used to calculate nrmse and mse
        self.calculated_nrmse = nrmse(Y_pred, Y_actual)
        self.calculated_mse = (Y_pred, Y_actual)

        return self.calculated_nrmse


    #override abstract Reset method
    def Reset_Reservoir(self):

        from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
        from sklearn.linear_model import Ridge
        initially_fixed_params = {'hidden_layer_size': self.reservoir_nodes,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'reservoir_activation': 'tanh',
                          'leakage': self.leakage_rate,
                          'bidirectional': False,
                          'k_rec': 10,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-5,
                          'random_state': 42,
                          'requires_sequence': False,
                          'spectral_radius': self.spectral_radius}

        self.esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params) 

        pass
