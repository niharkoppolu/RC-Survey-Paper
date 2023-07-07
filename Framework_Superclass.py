#Import ABC to allow for abstract classes, time for calculating training time

from abc import ABC, abstractmethod
import time #used to time code
import numpy as np
import statistics
import matplotlib.pyplot as plt



#Global Variables
Mackey_Glass_txt_FilePath = "Mackey_Glass.txt"


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Creating Data Set
def Data_Processor():
    #Reading in data as list of strings (each index is 1 line in the file)
    data_file = open(Mackey_Glass_txt_FilePath, "r")
    raw_data_strings = data_file.readlines()
    float_data = []

    for str_number in raw_data_strings:
        float_data.append(float(str_number.split("\n")[0]))

    #turning mackey glass array into numpy array
    mackey_glass_final = np.asarray(float_data).reshape(-1,1) #reservoirpy expects array to be in a specific shape
    print(mackey_glass_final.shape) #debug

    #Print Data Set as Graph
    
    plt.figure(figsize=(10, 3))
    plt.title("Mackey Glass Data")
    plt.xlabel("First 200 Timesteps (10000 total in dataset)")
    plt.plot(mackey_glass_final[0:200], label="Mackey_Glass_Data_Set", color="blue")
    plt.legend()
    plt.show()

    return mackey_glass_final

#Create global that Reservoir_Framework can use
set_mackey_glass = Data_Processor()

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Super class for all Reservoir Frameworks
class Reservoir_Framework(ABC): #must pass in ABC parameter to make class abstract

    def __init__(self, Set_Num_Nodes):
        self.training_timesteps = 100
        self.test_set_begin = 9000
        self.test_set_end = self.test_set_begin + 500
        
        #Mackey_Glass data is converted to np array stored in mackey_glass_final
        self.mackey_glass_final = set_mackey_glass

        self.X_train = self.mackey_glass_final[0:self.training_timesteps] 
        self.Y_train = self.mackey_glass_final[1:self.training_timesteps + 1] 

        self.calculated_nrmse = -1
        self.calculated_mse = -1

        self.reservoir_nodes = Set_Num_Nodes
        self.leakage_rate = 0.5
        self.spectral_radius = 0.9

        #used for data collection
        self.av_timestep = -1
        self.stand_dev_timestep = -1

        self.av_training_time = -1
        self.stand_dev_train_time = -1
        pass

#abstract methods

    #This function is ONLY meant to return the values of the Optimized hyperparams
    #It DOES NOT create a Reservoir Computer
    @abstractmethod
    def Optimize_Hyperparam(self):
        pass

    @abstractmethod
    def Train(self):
        pass

    @abstractmethod
    def Test(self):
        pass

    @abstractmethod
    def Reset_Reservoir(self):
        pass

#END of abstract methods
#_________________________________________________________________________________________________________________________________________________

#Non-Abstract methods


    #This function is used to find the minimum number of timesteps it takes to Train to 100% accuracy
    def Find_Min_Timesteps(self):
        
        lowest_possible_timesteps_flag = False #this is set to true when reaching lowest number of timesteps with 0.01 nrmse
        nrmse_0_01_flag = False #used to check nrmse of previous_timestep training was 0.01 when the current nrmse is greater

        previous_timesteps = -1 #prev. number of timesteps used to train previous reservoir

        while lowest_possible_timesteps_flag == False:
            self.Reset_Reservoir()
            self.Train()
            self.calculated_nrmse = self.Test()
            #DEBUG
            print("\nnrmse: ", self.calculated_nrmse)

            #if it hasn't reached required accuracy add more timesteps, OR if previous timesteps had reached required accuracy exit
            #if training timesteps is not
            if self.calculated_nrmse > 0.01 and self.training_timesteps < 2000:
                if nrmse_0_01_flag == True:
                    self.training_timesteps = previous_timesteps
                    lowest_possible_timesteps_flag = True
                    pass

                else:    
                    previous_timesteps = self.training_timesteps
                    self.training_timesteps = self.training_timesteps + 50
                    pass

                pass



            #if it has reached required accuracy check to see if less timesteps would still work 
            else:
                nrmse_0_01_flag = True
                previous_timesteps = self.training_timesteps
                self.training_timesteps = self.training_timesteps - 1
                pass

            #DEBUG
            """
            if self.calculated_nrmse < 0.1:
                lowest_possible_timesteps_flag = True
                print("\nnrmse: " + str(self.calculated_nrmse))
            """


            pass
        #will not be used in paper, but will be used by to determine if same timesteps for all RC 
        print("Personal Metric: The minimum no. of timesteps it takes to reach accuracy of 0.01 nrmse is: " + str(self.training_timesteps))
        return self.training_timesteps



#_________________________________________________________________________________________________________________________________________________
                

    #may implement in base  
    #Only call after completing Find_Min_Timesteps
    def Time_Training(self):

        #overall_training_time = timeit.timeit(stmt="self.Train()", globals=globals(), number=1)

        #used to time Training (not sure if I should be using this method)
        #link to methods for finding Execution Time: https://pynative.com/python-get-execution-time-of-program/
        
        
        #--------Monte Carlo Simulation for Calculating overall training time------#
        self.Reset_Reservoir()

        monte_carlo_sim_size = 10 #set to 1000 when generating data
        overall_training_time_array = [0] * monte_carlo_sim_size
        print(len(overall_training_time_array))

        i = 0
        while i < monte_carlo_sim_size:
            #May need to add Reset Call here
            start_time = time.process_time()
            self.Train()
            end_time = time.process_time()
            overall_training_time_array[i] = end_time - start_time
            i = i + 1
            pass
        #--------Monte Carlo Simulation for Calculating overall training time------#

        print("Training time of all trials before averaging: ",overall_training_time_array) #will comment out when generating data for paper
        print("Standard Deviation all trials: ", statistics.stdev(overall_training_time_array))
        
        train_step_array = np.array(overall_training_time_array) / self.training_timesteps
        self.av_training_time = sum(overall_training_time_array) / len(overall_training_time_array)


        print("\nRelevent: It takes ", self.av_training_time, " to train  to 0.01 nrmse (over ", self.training_timesteps, " timesteps.)\n")
        self.stand_dev_train_time = np.std(overall_training_time_array)
        print("std dev of Total Trainind Time: ", self.stand_dev_train_time)

        self.av_timestep = self.av_training_time / self.training_timesteps
        print("\n\nRelevent: Time per Train step: ", self.av_timestep, "\n")
        
        self.stand_dev_timestep = np.std(train_step_array)
        print("std dev of Time per Trainstep: ", self.stand_dev_timestep)
        

        return self.av_training_time

