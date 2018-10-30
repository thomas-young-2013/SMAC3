import traceback

from smac.configspace import Configuration
from smac.tae.execute_ta_run_aclib import StatusType, ExecuteTARunAClib
import automl_benchmarks.askl_benchmark

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteASKLRun(ExecuteTARunAClib):

    """
    Gets values from an ASKL surrogate
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surro = automl_benchmarks.askl_benchmark.CombinedBenchmark()
        self._max_fidelity = self.surro.get_meta_information()['steps'][-1]
        self._cutoff = self.surro.get_meta_information()['cutoff']

    def _call_ta(self,
                 config: Configuration,
                 instance: str,
                 instance_specific: str,
                 cutoff: float,
                 seed: int):
        stdout_ = 'Calling "self.surro.objective_function(%s, %s, %s)"' % (
            str(config), instance, str(self._max_fidelity))
        results = {'status': 'CRASHED',
                   'runtime': 99999999,
                   'cost': 99999999,
                   'info': instance_specific,
                   'seed': seed,
                   'cutoff': cutoff,
                   'instance_specific': instance_specific}
        stderr_ = ''
        try:
            instance = int(instance)
            res = self.surro.objective_function(config, instance, self._max_fidelity)
            results['runtime'] = res['cost'][0]
            results['cost'] = res['function_value'][0]
            stat = 'SUCCESS' if results['runtime'] < cutoff else 'TIMEOUT'
            results['status'] = stat
        except:
            stderr_ = traceback.format_exc()

        return results, stdout_, stderr_
