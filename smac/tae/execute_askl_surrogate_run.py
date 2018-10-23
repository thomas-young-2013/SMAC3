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
        #
        # log_func = print
        # log_func("##########################################################")
        # log_func("Statistics:")
        # log_func("#Incumbent changed: %d" %(self.stats.inc_changed - 1)) # first change is default conf
        # log_func("#Target algorithm runs: %d / %s" %(self.stats.ta_runs, str(9)))
        # log_func("#Configurations: %d" %(self.stats.n_configs))
        # # log_func("Used wallclock time: %.2f / %.2f sec " %(time.time() - self._start_time, self.__scenario.wallclock_limit))
        # log_func("Used target algorithm runtime: %.2f / %.2f sec" %(self.stats.ta_time_used, 9))
        # log_func(self.stats.is_budget_exhausted(), self.stats.get_remaining_ta_budget())
        # # self._logger.debug("Debug Statistics:")
        # # if self._n_calls_of_intensify > 0:
        # #     self._logger.debug("Average Configurations per Intensify: %.2f" %(self.stats._n_configs_per_intensify / self.stats._n_calls_of_intensify))
        # #     self._logger.debug("Exponential Moving Average of Configurations per Intensify: %.2f" %(self.stats._ema_n_configs_per_intensifiy))
        #
        # log_func("##########################################################")

        return results, stdout_, stderr_
