import logging
import os
import datetime
import time
import typing
import copy

import json
import multiprocessing

import numpy as np

from ConfigSpace.configuration_space import Configuration

from smac.tae.execute_ta_run_hydra import ExecuteTARunOld
from smac.tae.execute_ta_run_hydra import ExecuteTARun
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.optimizer.pSMAC import read
from smac.utils.io.output_directory import create_output_directory
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.utils.util_funcs import get_rng
from smac.utils.constants import MAXINT

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


def optimize(queue: multiprocessing.Queue,
             scenario: typing.Type[Scenario],
             tae: typing.Type[ExecuteTARun],
             rng: typing.Union[np.random.RandomState, int],
             output_dir: str,
             **kwargs) -> None:
    """
    Unbound method to be called in a subprocess

    Parameters
    ----------
    queue: multiprocessing.Queue
        incumbents (Configurations) of each SMAC call will be pushed to this queue
    scenario: Scenario
        smac.Scenario to initialize SMAC
    tae: ExecuteTARun
        Target Algorithm Runner (supports old and aclib format)
    rng: int/np.random.RandomState
        The randomState/seed to pass to each smac run
    output_dir: str
        The directory in which each smac run should write it's results

    """
    logger = logging.getLogger('Worker_%d' % multiprocessing.current_process().pid)
    logger.info('Worker_%d: Alive!', multiprocessing.current_process().pid)
    tae = tae(ta=scenario.ta, run_obj=scenario.run_obj)
    solver = SMAC(scenario=scenario, tae_runner=tae, rng=rng, **kwargs)
    solver.stats.start_timing()
    solver.stats.print_stats()

    logger.info('Worker_%d: Starting SMAC', multiprocessing.current_process().pid)
    incumbent = solver.solver.run()
    solver.stats.print_stats()
    logger.info('Worker_%d: SMAC done', multiprocessing.current_process().pid)

    if output_dir is not None:
        solver.solver.runhistory.save_json(
            fn=os.path.join(solver.output_dir, "runhistory.json")
        )
    logger.info('Worker_%d: Pushing to queue', multiprocessing.current_process().pid)
    queue.put((incumbent, solver.stats.get_savable_data()), block=False)
    queue.close()
    logger.info('Worker_%d: Pushed to queue', multiprocessing.current_process().pid)


class PSMAC(object):
    """
    Facade to use PSMAC

    Attributes
    ----------
    logger
    stats : Stats
        loggs information about used resources
    solver : SMBO
        handles the actual algorithm calls
    rh : RunHistory
        List with information about previous runs
    portfolio : list
        List of all incumbents

    """

    def __init__(self,
                 scenario: typing.Type[Scenario],
                 rng: typing.Optional[typing.Union[np.random.RandomState, int]] = None,
                 run_id: int = 1,
                 tae: typing.Type[ExecuteTARun] = ExecuteTARunOld,
                 shared_model: bool = True,
                 validate: bool = True,
                 n_optimizers: int = 2,
                 val_set: typing.Union[typing.List[str], None] = None,
                 n_incs: int = 1,
                 use_epm: bool = False,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_optimizers: int
            Number of optimizers to run in parallel per round
        rng: int/np.random.RandomState
            The randomState/seed to pass to each smac run
        run_id: int
            run_id for this hydra run
        tae: ExecuteTARun
            Target Algorithm Runner (supports old and aclib format as well as AbstractTAFunc)
        shared_model: bool
            Flag to indicate whether information is shared between SMAC runs or not
        validate: bool / None
            Flag to indicate whether to validate the found configurations or to use the SMAC estimates
            None => neither and return the full portfolio
        n_incs: int
            Number of incumbents to return (n_incs <= 0 ==> all found configurations)
        val_set: typing.List[str]
            List of instance-ids to validate on
        use_epm: bool
            Flag to determine if the validation uses real runs or EPM predictions

        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.scenario = scenario
        self.run_id, self.rng = get_rng(rng, run_id, logger=self.logger)
        self.kwargs = kwargs
        self.output_dir = None
        self.rh = RunHistory(average_cost)
        self._tae = tae
        self.tae = tae(ta=self.scenario.ta, run_obj=self.scenario.run_obj)
        if n_optimizers <= 1:
            self.logger.warning('Invalid value in %s: %d. Setting to 2', 'n_optimizers', n_optimizers)
        self.n_optimizers = max(n_optimizers, 2)
        self.validate = validate
        self.shared_model = shared_model
        self.n_incs = min(max(1, n_incs), self.n_optimizers)
        if val_set is None:
            self.val_set = scenario.train_insts
        else:
            self.val_set = val_set
        self.use_epm = use_epm
        self.stats = {
            'wallclock_time_used': 0,
            'cumulative':
                {
                    'ta_runs': 0,
                    'n_configs': 0,
                    'ta_time_used': 0,
                    'inc_changed': 0
                },
            'individual':
                {
                    'ta_runs': [],
                    'n_configs': [],
                    'ta_time_used': [],
                    'inc_changed': []
                }
        }
        self.validation_stats = {
            'ta_runs': 0,
            'n_configs': 0,
            'wallclock_time_used': 0,
            'ta_time_used': 0,
            'inc_changed': 0
        }

    def _update_stats(self, stats):
        self.stats['wallclock_time_used'] = max(self.stats['wallclock_time_used'],
                                                stats['wallclock_time_used'])
        for key in self.stats['cumulative']:
            self.stats['cumulative'][key] += stats[key]
            self.stats['individual'][key].append(stats[key])

    def save_stats(self, path):
        path = os.path.join(path, 'stats.json')
        self.logger.debug("Saving stats to %s", path)
        with open(path, 'w') as fh:
            json.dump({'stats': self.stats, 'validation': self.validation_stats}, fh, indent=4, sort_keys=True)

    def optimize(self):
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        incumbent(s) : Configuration / List[Configuration] / ndarray[Configuration]
            Incumbent / Portfolio of incumbents

        """
        # Setup output directory
        if self.output_dir is None:
            self.scenario.output_dir = "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
            if self.shared_model:
                self.scenario.shared_model = self.shared_model
        if self.scenario.input_psmac_dirs is None:
            self.scenario.input_psmac_dirs = os.path.sep.join((self.scenario.output_dir, 'run_*'))

        scen = copy.deepcopy(self.scenario)
        scen.output_dir_for_this_run = None
        scen.output_dir = None
        self.logger.info("+" * 120)
        self.logger.info("PSMAC run")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiprocessing part start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        q = multiprocessing.Queue(maxsize=-1)
        procs = []
        for p in range(self.n_optimizers):
            proc = multiprocessing.Process(target=optimize,
                                           args=(
                                               q,  # Output queue
                                               self.scenario,  # Scenario object
                                               self._tae,  # type of tae to run target with
                                               p,  # process_id (used in output folder name)
                                               self.output_dir,  # directory to create outputs in
                                           ),
                                           kwargs=self.kwargs)
            self.logger.info('Starting process: %d', p)
            proc.start()
            procs.append(proc)
        incs = np.empty((self.n_optimizers,), dtype=Configuration)
        idx = 0
        self.logger.info('Emptying Queue')
        while idx < self.n_optimizers:
            conf, stats = q.get()
            # self.logger.info('Stats: %s', str(stats))
            self.logger.info('Received Config')
            self._update_stats(stats)
            incs[idx] = conf
            idx += 1
        for proc in procs:
            self.logger.info('Joining process: %d', proc.pid)
            proc.join()
        self.logger.info('Queue empty')
        self.logger.info('Loading all runhistories')
        # reads in all runs, stores in self.rh, needed to estimate best config
        read(self.rh, self.scenario.input_psmac_dirs, self.scenario.cs, self.logger)
        q.close()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiprocessing part end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        if self.n_optimizers == self.n_incs:  # no validation necessary just return all incumbents
            return incs
        else:
            ids, _ = self.get_best_incumbents_ids(incs, self.validate)  # determine the best incumbents
            return incs[ids]

    def get_best_incumbents_ids(self, incs: typing.Union[typing.List[Configuration], np.ndarray],
                                validate: bool = True):
        """
        Determines the IDs and costs of the best configurations

        Parameters
        ----------
        incs : typing.List[Configuration]
            incumbents determined by all parallel SMAC runs
        validate: bool
            Flag to determine if the ids are based on configurator estimates or on validated data

        Returns
        -------
        list
            Id(s) of best configuration(s)
        dict
            Cost per incumbent in incs

        """
        rh = self.validate_incs(incs) if validate else self.rh
        mean_costs_conf, cost_per_config = self.get_mean_costs(incs, rh)
        ids = list(map(lambda x: x[0],
                       sorted(enumerate(mean_costs_conf), key=lambda y: y[1])))[:self.n_incs]
        return ids, cost_per_config

    @staticmethod
    def get_mean_costs(incs: typing.List[Configuration], new_rh: RunHistory):
        """
        Compute mean cost per instance

        Parameters
        ----------
        incs : typing.List[Configuration]
            incumbents determined by all parallel SMAC runs
        new_rh : RunHistory
            runhistory to determine mean performance

        Returns
        -------
        List[float] means
        Dict(Config -> Dict(inst_id(str) -> float))

        """
        config_cost_per_inst = {}
        results = []
        for incumbent in incs:
            cost_per_inst = new_rh.get_instance_costs_for_config(config=incumbent)
            config_cost_per_inst[incumbent] = cost_per_inst
            results.append(np.mean(list(cost_per_inst.values())))
        return results, config_cost_per_inst

    def validate_incs(self, incs: np.ndarray):
        """
        Validation
        """
        solver = SMAC(scenario=self.scenario, tae_runner=self.tae, rng=self.rng, run_id=MAXINT, **self.kwargs)
        solver.solver.runhistory = self.rh
        self.logger.info('*' * 120)
        self.logger.info('Validating')
        new_rh = solver.validate(config_mode=incs,
                                 instance_mode=self.val_set,
                                 repetitions=1,
                                 use_epm=self.use_epm,
                                 n_jobs=self.n_optimizers)
        self.validation_stats = solver.solver.intensifier.tae_runner.stats.get_savable_data()
        del solver
        return new_rh
