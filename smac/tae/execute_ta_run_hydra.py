import typing

from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run_aclib import ExecuteTARun
from smac.tae.execute_askl_surrogate_run import ExecuteASKLRun
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_func import ExecuteTAFuncArray
from smac.tae.execute_ta_run_old import StatusType

__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"


class ExecuteTARunHydra(ExecuteTARun):

    """Returns min(cost, cost_portfolio)
    """
    
    def __init__(self,
                 cost_oracle: typing.Mapping[str, float] = None,
                 tae: typing.Type[ExecuteTARun] = ExecuteTARunOld,
                 portfolio: typing.Union[None, dict] = None,
                 **kwargs):
        '''
            Constructor
            
            Arguments
            ---------
            cost_oracle: typing.Mapping[str,float]
                cost of oracle per instance
            tae: ExecuteTARun
                target algorithm evaluator
            portfolio: typing.Mapping[str,type.List[str]]
                Current portfolio cost per instance
        '''

        super().__init__(**kwargs)
        self.cost_oracle = cost_oracle
        self.port = portfolio
        self.logger.info('cost_oracle: %s', str(self.cost_oracle is not None))
        self.logger.info('portfolio: %s', str(self.port is not None))
        if self.cost_oracle:
            self.logger.info('HYDRA: using standard TAE')
        elif self.port:
            self.logger.info('HYDRA: using relaxed TAE')
        else:
            raise Exception('No information about Portfolio provided!')
        if tae is ExecuteASKLRun:
            self.runner = ExecuteASKLRun(**kwargs)
        elif tae is ExecuteTARunAClib:
            self.runner = ExecuteTARunAClib(**kwargs)
        elif tae is ExecuteTARunOld:
            self.runner = ExecuteTARunOld(**kwargs)
        elif tae is ExecuteTAFuncDict:
            self.runner = ExecuteTAFuncDict(**kwargs)
        elif tae is ExecuteTAFuncArray:
            self.runner = ExecuteTAFuncArray(**kwargs)
        else:
            raise Exception('TAE not supported')

    def run(self, **kwargs):
        """ see ~smac.tae.execute_ta_run.ExecuteTARunOld for docstring
        """

        status, cost, runtime, additional_info = self.runner.run(**kwargs)
        inst = kwargs["instance"]
        if self.cost_oracle:
            try:
                oracle_perf = self.cost_oracle[inst]
            except KeyError:
                oracle_perf = None
            if oracle_perf is not None:
                if self.run_obj == "runtime":
                    self.logger.debug("Portfolio perf: %f vs %f = %f", oracle_perf, runtime, min(oracle_perf, runtime))
                    runtime = min(oracle_perf, runtime)
                    cost = runtime
                else:
                    self.logger.debug("Portfolio perf: %f vs %f = %f", oracle_perf, cost, min(oracle_perf, cost))
                    cost = min(oracle_perf, cost)
                if oracle_perf < kwargs['cutoff'] and status is StatusType.TIMEOUT:
                    status = StatusType.SUCCESS
            else:
                self.logger.error("Oracle performance missing --- should not happen")
        elif self.port:
            port_perf = self.port[inst]
            val = runtime if self.run_obj == 'runtime' else cost
            relaxed_perf = sum(map(lambda x: min(x, val), port_perf)) / len(port_perf)
            self.logger.debug('Portfolio perf: %s vs %f = %f', str(port_perf), val, relaxed_perf)
            if relaxed_perf < kwargs['cutoff'] and status is StatusType.TIMEOUT:
                status = StatusType.SUCCESS
            if self.run_obj == 'runtime':
                runtime = relaxed_perf
            else:
                cost = relaxed_perf

        return status, cost, runtime, additional_info
