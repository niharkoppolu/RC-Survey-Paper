"""
Project: RC Frameworks: 1-Step Prediction Comparison

This program Completes a timing analysis of different RC Frameworks completing a 1-step prediction task using the Mackey Glass dataset.
The program is organized like so: 

1. Data_Processor function: function used to process the Mackey_Glass data and turn it into training and testing data that can be used by the RCs

2. Reservoir_Framework class: Abstract Base Class used as structure for child classes. Each child class implements a RC using a different framework. This class contains the implementation of:
    - Find_Min_Timesteps: used to find the minimum number of training timesteps it takes to reach 0.01 nrmse
    - Time_Training: used to find the amount of time it takes to train the RC to reach 0.01 nrmse


3. Each of the following is a child class used to implement RC with the framework specified. 
    - 3a. ReservoirPy
    - 3b. EasyESN (DOESNT WORK)
    - 3c. pyRCN


Tlab Meeting Notes:

1. - Use a monte carlo simulation to determine overall training time, time per timestep over 100's of runs: Complete

2. - Do same comparison with same task 1000 nodes instead of 100: as the reservoir grows should 

3. - After doing the 2 1-step pred with 100 and 1000 nodes, try more complex task 
"""

#Importing classes used to test different FRAMEWORKS

from ReservoirPy_class import ReservoirPy
from pyRCN_class import pyRCN


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#imports used by all classes (No Framework imports)
import matplotlib.pyplot as plt

#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#This function is used separately to Run Tests on Frameworks to determine length of Timesteps
def Time_Step_Tests(Framework_RC, Framework_Name):

    Res_Size = [100, 200, 300, 400]
    # add 100, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1000
    #do 100 montocarlo runs
    


    #Test Below
    #Res_Size = [100, 200]

    #Comprehensive_Framework_List = [[ReservoirPy(Res_Size[0]), ReservoirPy(Res_Size[1]), ReservoirPy(Res_Size[2]), ReservoirPy(Res_Size[3])],
    #                                [pyRCN(Res_Size[0]), pyRCN(Res_Size[1]), pyRCN(Res_Size[2]), pyRCN(Res_Size[3])]]

    Av_Timestep_List = []
    Timestep_Error = []

    Av_Time_List = []
    Total_Time_Error = []

    for res_size in Res_Size:
        
        print("\n\n\n\n_______________________________________________________________________________________________")
        print("_______________________________________________________________________________________________")
        print(res_size, " Node Reservoir:\n\n")
        reservoir = Framework_RC(res_size)
        reservoir.Find_Min_Timesteps()
        #link to methods for finding Execution Time: https://pynative.com/python-get-execution-time-of-program/
        reservoir.Time_Training()

        Av_Timestep_List.append(reservoir.av_timestep)
        Timestep_Error.append(reservoir.stand_dev_timestep)

        Av_Time_List.append(reservoir.av_training_time)
        Total_Time_Error.append(reservoir.stand_dev_train_time)
        pass


    #___________________________________________________________________________________________________________
    #Graph of Training Timesteps of different Reservoir Computer Sizes
    print("\n\n\n\n_______________________________________________________________________________________________")
    print("_______________________________________________________________________________________________")
    print("Graph 1: Training Timesteps\n\n")


    #https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.07-Error-Bars/ : used this for creating plot
    #____________________________Graph 1: Timestep of Dif. Reservoir Sizes_______
    fig, ax = plt.subplots()

    ax.errorbar(Res_Size, Av_Timestep_List,
                yerr=Timestep_Error,
                alpha=0.5,
                ecolor='black',
                capsize=10)

    ax.set_xlabel('Reservoir Size')
    ax.set_ylabel('Time per Trainstep (Sec')
    ax.set_title('ReservoirPy: Training Timestep Length')

    plt.savefig(Framework_Name + '_Graph 1:_time_step_plot.png')
    
    #____________________________Graph 2: Total Training Time of Dif. Reservoir Sizes_______
    print("Graph 2: Total Training Time\n\n")
    fig, ax = plt.subplots()

    ax.errorbar(Res_Size, Av_Time_List,
                yerr=Total_Time_Error,
                alpha=0.5,
                ecolor='black',
                capsize=10)


    ax.set_xlabel('Reservoir Size')
    ax.set_ylabel('Total Train Time (Sec)')
    ax.set_title('Graph 2 ReservoirPy: Total Training Time Length')


    plt.show
    plt.savefig(Framework_Name + '_Graph_2:_total-time_plot.png')
    pass


#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________
#Main Point of Execution

Framework_dict = {
  "ReservoirPy": ReservoirPy,
  "pyRCN": pyRCN,
}

print("\n\nStart ReservoirPy Tests\n\n")
#This completes reservoirpy tests
Time_Step_Tests(Framework_dict["ReservoirPy"], 'ReservoirPy')

#Need to Do some debugging of pyRCN code
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n_______________________________________________________________________________________________")
print("_______________________________________________________________________________________________")
print("\n\nStart pyRCN Tests\n\n")
Time_Step_Tests(Framework_dict["pyRCN"], 'pyRCN')

"""
#Testing pyRCN: Not done yet - Need to Find way how to set Reservoir sizes
print("\n\nTesting pyRCN\n\n")
pyRCN_test = pyRCN(100)
pyRCN_test.Train()
pyRCN_test.Test()
"""
