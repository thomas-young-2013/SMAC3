import logging
import os
import inspect

import numpy as np

from smac.configspace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.optimizer.ei_optimization import FiniteSearch


def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 5].
    """
    x1 = x["x1"]
    x2 = x["x2"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val


logger = logging.getLogger("Finite-example")
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output

cs = ConfigurationSpace()

x1 = UniformFloatHyperparameter("x1", -1, 1, default_value=0.0)
x2 = UniformFloatHyperparameter("x2", -1, 1, default_value=0.0)

cs.add_hyperparameters([x1,x2])

configs = [Configuration(cs, {"x1":x, "x2":x}) for x in np.linspace(-1,1,100)] 

# SMAC scenario oject
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                     "runcount-limit": 50,  # maximum number of function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     "memory_limit": 1024,   # adapt this to reasonable value for your hardware
                     })

# use FiniteSearch to build challenger list given a finite list of configurations
fs = FiniteSearch(config_space=cs, list_of_configs=configs)

# To optimize, we pass the function to the SMAC-object
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=rosenbrock_2d,
            acquisition_function_optimizer=fs)

# if you want to use SMAC, patch the acquisition func
# into FiniteSearch
# if you want to run ROAR, comment the next two lines out
smac.solver.acq_optimizer.acquisition_function = \
    smac.solver.acquisition_func

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
print("Value for default configuration: %.2f" % (def_value))

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
print("Optimized Value: %.2f" % (inc_value))
