from Framework_Superclass import Reservoir_Framework
from reservoirpy.observables import mse, nrmse, rmse

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 2: pyESN
class pyESN(Reservoir_Framework):
    
    def __init__(self):
        import pyESN
        super().__init__() #must call init function of superclass
        
        self.esn_model = pyESN.ESN(n_inputs = 1, n_outputs = 1, n_reservoir = self.reservoir_nodes, 
            spectral_radius = self.spectral_radius, random_state=42)

        


        pass

    #override abstract Train method
    def Train(self):
        
        self.X_train = self.mackey_glass_final[0:self.training_timesteps] #must add self for class variables
        self.Y_train = self.mackey_glass_final[1:self.training_timesteps + 1] 

        #pred_training = esn.fit(np.ones(trainlen),data[:trainlen]) 
        #This is used to predict entire Mackey Glass Series.
        #Ones are inserted as input because the input doesn't matter, the goal is to learn the series
        
        pred_training = self.esn_model.fit(self.X_train, self.Y_train) #This is used for 1-step timestep prediction
        #pred_training variable isn't used for much
        pass

    
    #override abstract Test method
    def Test(self):

        Y_pred = self.esn_model.predict(self.mackey_glass_final[self.test_set_begin : self.test_set_end])
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
        import pyESN

        self.esn_model = pyESN.ESN(n_inputs = 1, n_outputs = 1, n_reservoir = self.reservoir_nodes, 
            spectral_radius = self.spectral_radius, random_state=42)
        pass
