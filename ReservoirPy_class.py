from Framework_Superclass import Reservoir_Framework
from reservoirpy.observables import mse, nrmse, rmse, rsquare
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Framework 1: ReservoirPy (import of framework is placed in init)
class ReservoirPy(Reservoir_Framework): #superclass containing in brackets

    def __init__(self, Set_Num_Nodes):
        
        super().__init__(Set_Num_Nodes) #must call init function of superclass
        
        self.best_params = self.Optimize_Hyperparam()

        self.reservoir = Reservoir(self.reservoir_nodes, 
                                    sr=self.best_params['sr'], 
                                    lr=self.best_params['lr'], 
                                    inut_scaling=self.best_params['iss'], 
                                    seed=1234)
        
        self.readout = Ridge(ridge=1e-7)
        self.esn_model = self.reservoir >> self.readout #connects reservoir and ridge

        pass

    def Optimize_Hyperparam(self):

        #The objective function is used by ReservoirPy research function to determine hyperparameters
        def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):
            # You can access anything you put in the config 
            # file from the 'config' parameter.
            instances = config["instances_per_trial"]
            
            # The seed should be changed across the instances, 
            # to be sure there is no bias in the results 
            # due to initialization.
            variable_seed = seed 

            losses = []; r2s = [];
            for n in range(instances):
                # Build your model given the input parameters
                reservoir = Reservoir(N, 
                                    sr=sr, 
                                    lr=lr, 
                                    inut_scaling=iss, 
                                    seed=variable_seed)
                
                readout = Ridge(ridge=ridge)

                model = reservoir >> readout

                x_test = self.mackey_glass_final[self.test_set_begin : self.test_set_end]
                y_test = self.mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1] 

                # Train your model and test your model.
                predictions = model.fit(self.X_train, self.Y_train) \
                                .run(x_test)
                
                loss = nrmse(y_test, predictions, norm_value=np.ptp(self.X_train))
                r2 = rsquare(y_test, predictions)
                
                # Change the seed between instances
                variable_seed += 1
                
                losses.append(loss)
                r2s.append(r2)
                pass #end of FOR

            # Return a dictionnary of metrics. The 'loss' key is mandatory when
            # using hyperopt.
            return {'loss': np.mean(losses),
                    'r2': np.mean(r2s)}
            pass #End of objective function
        #END of OBJECTIVE func

        hyperopt_config = {
            "exp": f"hyperopt-multiscroll", # the experimentation name
            "hp_max_evals": 200,             # the number of differents sets of parameters hyperopt has to try
            "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
            "seed": 42,                      # the random state seed, to ensure reproducibility
            "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters

            "hp_space": {                    # what are the ranges of parameters explored
                "N": ["choice", self.reservoir_nodes],             # the number of neurons is fixed to 500
                "sr": ["loguniform", 1e-2, 2],   # the spectral radius is log-uniformly distributed between 1e-2 and 2
                "lr": ["loguniform", 1e-3, 1],  # idem with the leaking rate, from 1e-3 to 1
                "iss": ["loguniform",1e-2, 2],           # the input scaling is fixed #MAY CAUSE ERROR
                "ridge": ["choice", 1e-7],        # and so is the regularization parameter.
                "seed": ["choice", 1234]          # an other random seed for the ESN initialization
            }
        }  


        #__________ Not Sure if Necessary
        import json
        # we precautionously save the configuration in a JSON file
        # each file will begin with a number corresponding to the current experimentation run number.
        with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
            json.dump(hyperopt_config, f)
        #__________

        from reservoirpy.hyper import research
        #creating 'dataset' which is used by the research function
        x_test = self.mackey_glass_final[self.test_set_begin : self.test_set_end]
        y_test = self.mackey_glass_final[self.test_set_begin + 1 : self.test_set_end + 1] 
        dataset = ((self.X_train, self.Y_train), (x_test, y_test))

        #This function completes the Hyperoptimization task 
        self.best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
        return self.best[0] #This contains a dictionary of values that were hyperoptimized 
         
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

