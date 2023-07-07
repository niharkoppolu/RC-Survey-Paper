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

        #1. DEFINE INITIAL ESN MODEL  
        # Only fixing size of hidden layer
        initially_fixed_params = {'hidden_layer_size': self.reservoir_nodes}
                          #'input_activation': 'identity',
                          #'bias_scaling': 0.0,
                          #'reservoir_activation': 'tanh',
                          #'leakage': self.leakage_rate,
                          #'bidirectional': False,
                          #'k_rec': 10,
                          #'wash_out': 0,
                          #'continuation': False,
                          #'alpha': 1e-5,
                          #'random_state': 42,
                          #'requires_sequence': False,
                          #'spectral_radius': self.spectral_radius}
              
        self.esn = ESNRegressor(bias_scaling=0, spectral_radius=0, leakage=1, **initially_fixed_params)
        #old self.esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params) 
        #old self.esn = ESNRegressor()
        
        #2. Calling Optimization function
        print("\n\nCalling optimization function\n\n")
        self.best_params = self.Optimize_Hyperparam()
        print("Hyperparameter optimization complete")
        self.esn = ESNRegressor(regressor=Ridge(), **self.best_params) 


        pass


    #override abstract Optimize_Hyperparam method
    def Optimize_Hyperparam(self):

        from sklearn.metrics import make_scorer
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV)
        from scipy.stats import uniform
        from pyrcn.model_selection import SequentialSearchCV
        from pyrcn.datasets import mackey_glass
        from pyrcn.echo_state_network import ESNRegressor
        from sklearn.linear_model import Ridge


        #dataset already loaded from Superclass inheritance


        #1. Define optimization workflow
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        step_1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                        'spectral_radius': uniform(loc=0, scale=2)}
        kwargs_1 = {'n_iter': 200, 'n_jobs': -1, 'scoring': scorer, 
                    'cv': TimeSeriesSplit()}

        step_2_params = {'leakage': [0.4, 0.7, 0.9]} #Used for testing
        #step_2_params = {'leakage': [0.2, 0.4, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]} #altered the leakage values being tested
        kwargs_2 = {'verbose': 5, 'scoring': scorer, 'n_jobs': -1,
                    'cv': TimeSeriesSplit()}

        searches = [('step1', RandomizedSearchCV, step_1_params, kwargs_1),
                    ('step2', GridSearchCV, step_2_params, kwargs_2)]

        #2. PERFORM THE SEARCH
        sequential_search = SequentialSearchCV(self.esn, searches).fit(self.X_train.reshape(-1, 1), self.Y_train)
        best_params = sequential_search.best_estimator_.get_params()
        return best_params

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

        #RESETS WITH OPTIMIZED Hyperparameters
        self.esn = ESNRegressor(regressor=Ridge(), **self.best_params) 

        #old code: UNOPTIMIZED RC"
        """
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

        #self.esn = ESNRegressor(regressor=Ridge(), **initially_fixed_params) 
        """
        pass
