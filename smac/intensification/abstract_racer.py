import logging
import typing
import time
from collections import OrderedDict

import numpy as np

from smac.optimizer.objective import sum_cost

from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger

# (for now) to avoid cyclic imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import smac.optimizer.smbo.SMBO

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class AbstractRacer(object):
    """
    Base class for all racing methods

    The "intensification" is designed to be spread across multiple ``eval_config()`` runs.
    This is to facilitate on-demand configuration sampling if multiple configurations are required,
    like Successive Halving or Hyperband.

    **Note: Do not use directly**

    Parameters
    ----------
    tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
        target algorithm run executor
    stats: Stats
        stats object
    traj_logger: TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : int
        runtime cutoff of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    minR : int
        Minimum number of run per config (summed over all calls to
        intensify).
    maxR : int
        Maximum number of runs per config (summed over all calls to
        intensifiy).
    adaptive_capping_slackfactor: float
        slack factor of adpative capping (factor * adpative cutoff)
    """

    def __init__(self, tae_runner: ExecuteTARun,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[int] = None,
                 deterministic: bool = False,
                 run_obj_time: bool = True,
                 minR: int = 1,
                 maxR: int = 2000,
                 adaptive_capping_slackfactor: float = 1.2,
                 **kwargs):

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.stats = stats
        self.traj_logger = traj_logger
        self.rs = rng

        # scenario info
        self.cutoff = cutoff
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time
        self.tae_runner = tae_runner

        self.adaptive_capping_slackfactor = adaptive_capping_slackfactor

        self.minR = minR
        self.maxR = maxR

        # instances
        if instances is None:
            instances = []
        # removing duplicates in the user provided instances
        self.instances = list(OrderedDict.fromkeys(instances))
        if instance_specifics is None:
            self.instance_specifics = {}
        else:
            self.instance_specifics = instance_specifics

        # general attributes
        self._min_time = 10 ** -5
        self._num_run = 0
        self._chall_indx = 0
        self._ta_time = 0

        # attributes for sampling next configuration
        self.repeat_configs = True
        # to sample fresh configs from EPM / mark the end of an iteration
        self.iteration_done = False

    def eval_challenger(self, challenger: Configuration,
                        incumbent: Configuration,
                        run_history: RunHistory,
                        aggregate_func: typing.Callable,
                        time_bound: float = float(MAXINT),
                        log_traj: bool = True) -> typing.Tuple[Configuration, float]:

        raise NotImplementedError()

    def get_next_challenger(self, challengers: typing.Optional[typing.List[Configuration]],
                            chooser: typing.Optional['smac.optimizer.smbo.SMBO'],
                            run_history: RunHistory,
                            repeat_configs: bool = True) -> typing.Optional[Configuration]:
        """
        Abstract method for choosing the next challenger, to allow for different selections across intensifiers
        uses ``_next_challenger()`` by default

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        chooser : 'smac.optimizer.smbo.SMBO'
            optimizer that generates next configurations to use for racing
        run_history : RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        typing.Optional[Configuration]
            next configuration to evaluate
        """
        challenger = self._next_challenger(challengers=challengers,
                                           chooser=chooser,
                                           run_history=run_history,
                                           repeat_configs=repeat_configs)
        return challenger

    def _next_challenger(self, challengers: typing.Optional[typing.List[Configuration]],
                         chooser: typing.Optional['smac.optimizer.smbo.SMBO'],
                         run_history: RunHistory,
                         repeat_configs: bool = True) -> typing.Optional[Configuration]:
        """ Retuns the next challenger to use in intensification
        If challenger is None, then optimizer will be used to generate the next challenger

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        chooser : smac.optimizer.smbo.SMBO
            a sampler that generates next configurations to use for racing
        run_history : RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        Configuration
            next challenger to use
        """
        start_time = time.time()

        used_configs = set(run_history.get_all_configs())

        if challengers:
            # iterate over challengers provided
            self.logger.debug("Using challengers provided")
            chall_gen = (c for c in challengers)
        elif chooser:
            # generating challengers on-the-fly if optimizer is given
            self.logger.debug("Generating new challenger from optimizer")
            chall_gen = chooser.choose_next()
        else:
            self.logger.debug("No configurations/chooser provided. Cannot generate challenger!")
            return None

        self.logger.debug('Time to select next challenger: %.4f' % (time.time() - start_time))

        # select challenger from the generators
        for challenger in chall_gen:
            if repeat_configs:  # repetitions allowed
                return challenger
            else:               # select only a unique challenger
                if challenger not in used_configs:
                    return challenger

        self.logger.debug("No valid challenger was generated!")
        return None

    def get_num_iterations(self) -> int:
        """
        helper method to get number of completed racing iterations
        """
        raise NotImplementedError()

    def _adapt_cutoff(self, challenger: Configuration,
                      incumbent: Configuration,
                      run_history: RunHistory,
                      inc_sum_cost: float) -> float:
        """Adaptive capping:
        Compute cutoff based on time so far used for incumbent
        and reduce cutoff for next run of challenger accordingly

        !Only applicable if self.run_obj_time

        !runs on incumbent should be superset of the runs performed for the
         challenger

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        run_history : RunHistory
            Stores all runs we ran so far
        inc_sum_cost: float
            Sum of runtimes of all incumbent runs

        Returns
        -------
        cutoff: float
            Adapted cutoff
        """

        if not self.run_obj_time:
            return self.cutoff

        curr_cutoff = self.cutoff if self.cutoff is not None else np.inf

        # cost used by challenger for going over all its runs
        # should be subset of runs of incumbent (not checked for efficiency
        # reasons)
        chall_inst_seeds = run_history.get_runs_for_config(challenger)
        chal_sum_cost = sum_cost(config=challenger,
                                 instance_seed_budget_keys=chall_inst_seeds,
                                 run_history=run_history)
        cutoff = min(curr_cutoff,
                     inc_sum_cost * self.adaptive_capping_slackfactor -
                     chal_sum_cost
                     )
        return cutoff

    def _compare_configs(self, incumbent: Configuration,
                         challenger: Configuration,
                         run_history: RunHistory,
                         aggregate_func: typing.Callable,
                         log_traj: bool = True) -> typing.Optional[Configuration]:
        """
        Compare two configuration wrt the runhistory and return the one which
        performs better (or None if the decision is not safe)

        Decision strategy to return x as being better than y:
            1. x has at least as many runs as y
            2. x performs better than y on the intersection of runs on x and y

        Implicit assumption:
            Challenger was evaluated on the same instance-seed pairs as
            incumbent

        Parameters
        ----------
        incumbent: Configuration
            Current incumbent
        challenger: Configuration
            Challenger configuration
        run_history: RunHistory
            Stores all runs we ran so far
        aggregate_func: typing.Callable
            Aggregate performance across instances
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        None or better of the two configurations x,y
        """

        inc_runs = run_history.get_runs_for_config(incumbent)
        chall_runs = run_history.get_runs_for_config(challenger)
        to_compare_runs = set(inc_runs).intersection(chall_runs)

        # performance on challenger runs
        chal_perf = aggregate_func(challenger, run_history, to_compare_runs)
        inc_perf = aggregate_func(incumbent, run_history, to_compare_runs)

        # Line 15
        if chal_perf > inc_perf and len(chall_runs) >= self.minR:
            # Incumbent beats challenger
            self.logger.debug("Incumbent (%.4f) is better than challenger "
                              "(%.4f) on %d runs." %
                              (inc_perf, chal_perf, len(chall_runs)))
            return incumbent

        # Line 16
        if not set(inc_runs) - set(chall_runs):

            # no plateau walks
            if chal_perf >= inc_perf:
                self.logger.debug("Incumbent (%.4f) is at least as good as the "
                                  "challenger (%.4f) on %d runs." %
                                  (inc_perf, chal_perf, len(chall_runs)))
                return incumbent

            # Challenger is better than incumbent
            # and has at least the same runs as inc
            # -> change incumbent
            n_samples = len(chall_runs)
            self.logger.info("Challenger (%.4f) is better than incumbent (%.4f)"
                             " on %d runs." % (chal_perf, inc_perf, n_samples))
            # Show changes in the configuration
            params = sorted([(param, incumbent[param], challenger[param])
                             for param in challenger.keys()])
            self.logger.info("Changes in incumbent:")
            for param in params:
                if param[1] != param[2]:
                    self.logger.info("  %s : %r -> %r" % param)
                else:
                    self.logger.debug("  %s remains unchanged: %r" %
                                      (param[0], param[1]))

            if log_traj:
                self.stats.inc_changed += 1
                self.traj_logger.add_entry(train_perf=chal_perf,
                                           incumbent_id=self.stats.inc_changed,
                                           incumbent=challenger)
            return challenger

        # undecided
        return None
