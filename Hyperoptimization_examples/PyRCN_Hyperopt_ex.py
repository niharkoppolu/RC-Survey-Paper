print("Importing all necessary packages\n")
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import (RandomizedSearchCV,
                                     GridSearchCV)
from scipy.stats import uniform
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.datasets import mackey_glass
from pyrcn.echo_state_network import ESNRegressor
print("Completed all imports")

print("\n\nPyRCN framework: Starting Hyperparameter optimization Algorithm\n")


print("1. Loading dataset\n")
# Load the dataset
X, y = mackey_glass(n_timesteps=5000)
X_train, X_test = X[:1900], X[1900:]
y_train, y_test = y[:1900], y[1900:]


print("2. Defining initial ESN model\n")
# Define initial ESN model
esn = ESNRegressor(bias_scaling=0, spectral_radius=0, leakage=1)


print("3. Defining the parametes in the 2 different searches that will be used to determine the optimized parameters\n")
# Define optimization workflow
scorer = make_scorer(mean_squared_error, greater_is_better=False)
step_1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                 'spectral_radius': uniform(loc=0, scale=2)}
kwargs_1 = {'n_iter': 200, 'n_jobs': -1, 'scoring': scorer, 
            'cv': TimeSeriesSplit()}
step_2_params = {'leakage': [0.2, 0.4, 0.7, 0.9, 1.0]}
kwargs_2 = {'verbose': 5, 'scoring': scorer, 'n_jobs': -1,
            'cv': TimeSeriesSplit()}

searches = [('step1', RandomizedSearchCV, step_1_params, kwargs_1),
            ('step2', GridSearchCV, step_2_params, kwargs_2)]

# Perform the search
print("4. Performing the search to optimize the parameters\n\n")
esn_opti = SequentialSearchCV(esn, searches).fit(X_train.reshape(-1, 1), y_train)
print("Hyperparameter optimization is COMPLETE")
print(esn_opti)