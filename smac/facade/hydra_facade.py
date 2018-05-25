import logging
import os
import datetime
import time
import typing
import copy
from collections import defaultdict

import pickle
from functools import partial

import numpy as np

from ConfigSpace.configuration_space import Configuration

from smac.tae.execute_ta_run_hydra import ExecuteTARunHydra
from smac.tae.execute_ta_run_hydra import ExecuteTARunOld
from smac.tae.execute_ta_run_hydra import ExecuteTARun
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.psmac_facade import PSMAC
from smac.utils.io.output_directory import create_output_directory
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.utils.util_funcs import get_rng
from smac.utils.constants import MAXINT
from smac.optimizer.pSMAC import read
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4LogCost
from smac.tae.execute_ta_run import StatusType
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.utils.util_funcs import get_types
from smac.configspace.util import convert_configurations_to_array

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Hydra(object):
    """
    Facade to use Hydra default mode

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
                 n_iterations: int,
                 val_set: str='train',
                 incs_per_round: int=1,
                 n_optimizers: int=1,
                 rng: typing.Optional[typing.Union[np.random.RandomState, int]]=None,
                 run_id: int=1,
                 tae: typing.Type[ExecuteTARun]=ExecuteTARunOld,
                 use_epm: bool=False,
                 mode: str='standard',
                 max_size: int=0,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_iterations: int,
            number of Hydra iterations
        val_set: str
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        incs_per_round: int
            Number of incumbents to keep per round
        n_optimizers: int
            Number of optimizers to run in parallel per round
        rng: int/np.random.RandomState
            The randomState/seed to pass to each smac run
        run_id: int
            run_id for this hydra run
        tae: ExecuteTARun
            Target Algorithm Runner (supports old and aclib format as well as AbstractTAFunc)
        use_epm: bool
            Flag to determine if the validation uses real runs or EPM predictions
        mode: str
            Determines how to construct the portfolio
            mip -> Hydra MIP: From multiple runs choose the k with the best SMAC estimate and then validate those k
            contribution -> At each iteration determine how much the configuration contributes to the oracle. Keep at
                            most k
            "else" -> Keep the k (estimated/validated) best configurations in each round
        max_size: int
            Maximum portfolio size. size <= 0 ==> unbounded portfolio size

        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.n_iterations = n_iterations
        self.scenario = scenario
        self.run_id, self.rng = get_rng(rng, run_id, self.logger)
        self.kwargs = kwargs
        self.output_dir = scenario.output_dir if not scenario.output_dir.startswith('smac') else None
        self.top_dir = None
        self.solver = None
        self.portfolio = None
        self.model = None
        self.runhistory2epm = None
        self.rh = RunHistory(average_cost)
        self._tae = tae
        self.tae = tae(ta=self.scenario.ta, run_obj=self.scenario.run_obj)
        if incs_per_round <= 0:
            self.logger.warning('Invalid value in %s: %d. Setting to 1', 'incs_per_round', incs_per_round)
        self.incs_per_round = max(incs_per_round, 1)
        if n_optimizers <= 0:
            self.logger.warning('Invalid value in %s: %d. Setting to 1', 'n_optimizers', n_optimizers)
        self.n_optimizers = max(n_optimizers, 1)
        self.val_set = self._get_validation_set(val_set)
        self.cost_per_inst = {}
        self.optimizer = None
        self.portfolio_cost = None
        self.use_epm = use_epm
        self.candidate_configs_cost_per_inst = {}
        self.mode = mode
        self.max_size = max_size if max_size >= 1 else MAXINT
        self.dequeued = []

    def _setup_model(self, scenario):
        """
        Setup for model to fill missing instance performance when using validation set
        """
        self.logger.info('Setting up model')
        num_params = len(scenario.cs.get_hyperparameters())
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        model = RandomForestWithInstances(types=types,
                                          bounds=bounds,
                                          instance_features=scenario.feature_array,
                                          seed=self.rng.randint(MAXINT),
                                          pca_components=scenario.PCA_DIM,
                                          unlog_y=scenario.run_obj == "runtime",
                                          num_trees=scenario.rf_num_trees,
                                          do_bootstrapping=scenario.rf_do_bootstrapping,
                                          ratio_features=scenario.rf_ratio_features,
                                          min_samples_split=scenario.rf_min_samples_split,
                                          min_samples_leaf=scenario.rf_min_samples_leaf,
                                          max_depth=scenario.rf_max_depth)
        if scenario.run_obj == "runtime":

            # if we log the performance data,
            # the RFRImputator will already get
            # log transform data from the runhistory
            cutoff = np.log10(scenario.cutoff)
            threshold = np.log10(scenario.cutoff *
                                 scenario.par_factor)

            imputor = RFRImputator(rng=self.rng,
                                   cutoff=cutoff,
                                   threshold=threshold,
                                   model=model,
                                   change_threshold=0.01,
                                   max_iter=2)

            runhistory2epm = RunHistory2EPM4LogCost(
                scenario=scenario, num_params=num_params,
                success_states=[StatusType.SUCCESS, ],
                impute_censored_data=True,
                impute_state=[StatusType.CAPPED, ],
                imputor=imputor)

        elif scenario.run_obj == 'quality':
            runhistory2epm = RunHistory2EPM4Cost(scenario=scenario, num_params=num_params,
                                                 success_states=[
                                                     StatusType.SUCCESS,
                                                     StatusType.CRASHED],
                                                 impute_censored_data=False, impute_state=None)
        else:
            raise ValueError('Not supported')
        self.model = model
        self.runhistory2epm = runhistory2epm

    def _get_validation_set(self, val_set: str, delete: bool=True) -> typing.List[str]:
        """
        Create small validation set for hydra to determine incumbent performance

        Parameters
        ----------
        val_set: str
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        delete: bool
            Flag to delete all validation instances from the training set

        Returns
        -------
        val: typing.List[str]
            List of instance-ids to validate on

        """
        if val_set == 'none':
            return None
        if val_set == 'train':
            return self.scenario.train_insts
        elif val_set[:3] != 'val':
            self.logger.warning('Can not determine validation set size. Using full training-set!')
            return self.scenario.train_insts
        else:
            size = int(val_set[3:])/100
            if size <= 0 or size >= 1:
                raise ValueError('X invalid in valX, should be between 0 and 1')
            insts = np.array(self.scenario.train_insts)
            # just to make sure this also works with the small example we have to round up to 3
            size = max(np.floor(insts.shape[0] * size).astype(int), 3)
            ids = np.random.choice(insts.shape[0], size, replace=False)
            val = insts[ids].tolist()
            if delete:
                self.scenario.train_insts = np.delete(insts, ids).tolist()
            self._setup_model(self.scenario)
            return val

    def get_contribution(self, portfolio: typing.List[Configuration], candidates: typing.List[Configuration]):
        """
        Compute the contribution for candidates given a portfolio in which they could be integrated

        Parameters
        ----------
        portfolio: typing.List[Configuration]
            Already existing portfolio
        candidates: typing.List[Configuration]
            Potential candidates to extend portfolio

        Returns
        -------
        typing.List[typing.Tuple[Configuration, float]]
            List of tuples containing the contribution of each candidate configuration

        """
        contribution = defaultdict(int)
        contribution_improvement = defaultdict(float)
        for instance in self.val_set:
            best = np.inf
            best_c = None
            prev_best = None
            for config in portfolio:
                if self.candidate_configs_cost_per_inst[config][instance] < best:
                    if best_c:
                        prev_best = best
                    best = self.candidate_configs_cost_per_inst[config][instance]
                    if best >= self.scenario.cutoff:
                        best = self.scenario.cutoff * self.scenario.par_factor
                    best_c = config

            contribution[best_c] += 1
            # For stochastic update of the mean
            contribution_improvement[best_c] = \
                contribution_improvement[best_c] * (contribution[best_c] - 1) / contribution[best_c]
            if prev_best:
                contribution_improvement[best_c] += prev_best - best
            else:
                contribution_improvement[best_c] += self.scenario.cutoff * self.scenario.par_factor - best
        self.logger.info(';,.,;'*24)
        self.logger.info('Contributions: ')
        _sum = np.sum(list(contribution_improvement.values()))
        results = []
        for config in candidates:
            if config in contribution:
                weighted_contribution = (contribution_improvement[config] / _sum) * contribution[config]
            else:
                weighted_contribution = 0
            self.logger.info(config)
            # contributes to solving #instances
            self.logger.info('%3d, %6.3f', contribution[config],
                             weighted_contribution)
            self.logger.info(' ')
            results.append((config, weighted_contribution))
        self.logger.info(';,.,;'*24)
        return results

    def get_contributing_configurations(self,
                                        portfolio: typing.List[Configuration],
                                        candidates: typing.List[Configuration]):
        """
        Construct portfolio only from configurations that contribute to the overall improvement

        Parameters
        ----------
        portfolio: typing.List[Configuration]
            Already existing portfolio
        candidates: typing.List[Configuration]
            Potential candidates to extend portfolio

        Returns
        -------
        list
            Configurations (all candidates) sorted by their contribution to the oracle

        """
        # portfolio = candidates => we want to find the best portfolio from all possible candidates
        results = self.get_contribution(portfolio, candidates)
        results_ids = list(map(lambda x: x[0], enumerate(sorted(results, key=lambda y: y[1], reverse=True))))
        results = np.array(results)[results_ids]
        stop = np.argmin(results[:, 1])
        stop = np.min((self.max_size, stop))
        return results[:, 0][:stop]

    def predict_missing_data(self, cost_per_inst: typing.Dict[str, float],
                             config: Configuration,
                             fit: bool=False) -> typing.Dict[str, float]:
        """
        For instances that were not validated, predict the missing performance values.

        Parameters
        ----------
        cost_per_inst: typing.Dict[str, float]
            Set to validate incumbent(s) on. [train, valX].
            train => whole training set,
            valX => train_set * 100/X where X in (0, 100)
        config: Configuration
            Flag to delete all validation instances from the training set
        fit: bool

        Returns
        -------
        typing.Dict[str, float]
            Dict(Config -> Dict(inst_id(str) -> float))

        """
        if len(self.val_set) != len(self.scenario.train_insts):
            if fit:
                X, y = self.runhistory2epm.transform(self.rh)
                self.model.train(X, y)
            for inst in self.scenario.train_insts:
                if inst not in self.val_set and inst not in cost_per_inst:
                    pred = self.model.predict(
                        np.array([
                            np.hstack([
                                convert_configurations_to_array([config])[0],
                                self.scenario.feature_dict[inst]]
                            )]))[0].flatten()[0]
                    if self.scenario.run_obj == "runtime":
                        pred = np.power(10, pred)
                    cost_per_inst[inst] = pred
        return cost_per_inst

    def optimize(self) -> typing.List[Configuration]:
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        portfolio : typing.List[Configuration]
            Portfolio of found configurations

        """
        # Setup output directory
        self.portfolio = []
        portfolio_cost = np.inf
        if self.output_dir is None:
            self.top_dir = "hydra-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
            self.scenario.output_dir = os.path.join(self.top_dir, "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
        else:
            self.top_dir = self.output_dir
            self.scenario.output_dir = os.path.join(self.top_dir, "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)

        scen = copy.deepcopy(self.scenario)
        scen.output_dir_for_this_run = None
        scen.output_dir = None
        # parent process SMAC only used for validation purposes
        self.solver = SMAC(scenario=scen, tae_runner=self.tae, rng=self.rng, run_id=self.run_id, **self.kwargs)
        for i in range(self.n_iterations):
            self.logger.info("="*120)
            self.logger.info("Hydra (%s) Iteration: %d", self.mode, (i + 1))

            self.optimizer = PSMAC(
                scenario=self.scenario,
                run_id=self.run_id,
                rng=self.rng,
                tae=self._tae if i == 0 else partial(ExecuteTARunHydra, cost_oracle=self.cost_per_inst),
                shared_model=False,
                validate=True if self.val_set else False,
                n_optimizers=self.n_optimizers,
                val_set=self.val_set,
                n_incs=self.n_optimizers,  # return all configurations (unvalidated)
                use_epm=self.use_epm,
                **self.kwargs
            )
            self.optimizer.output_dir = self.output_dir
            incs = self.optimizer.optimize()  # all incumbents out of all runs
            val = True if self.val_set else False
            val = False if self.mode == 'mip' else val
            to_keep_ids, cost_per_conf = self.optimizer.get_best_incumbents_ids(incs, val)
            self.candidate_configs_cost_per_inst = {**self.candidate_configs_cost_per_inst,
                                                    **cost_per_conf}
            config_cost_per_inst = {}
            if self.mode == 'contribution':
                incs = self.get_contributing_configurations(list(self.candidate_configs_cost_per_inst.keys()),
                                                            list(self.candidate_configs_cost_per_inst.keys()))
                self.portfolio = []  # reset the portfolio as incs contain all needed configurations!
                cost_per_conf = self.candidate_configs_cost_per_inst
            elif self.mode == 'mip':
                incs = incs[to_keep_ids][:self.incs_per_round]  # determine k best incumbents on SMAC estimates
                _, cost_per_conf = self.optimizer.get_best_incumbents_ids(incs, True)  # validate only those incumbents
            elif self.mode == 'rr':
                space = self.max_size - (len(self.portfolio) + len(incs))
                if self.dequeued:
                    deq = np.array(self.dequeued)
                    self.dequeued = []
                    incs = self.get_contributing_configurations(self.portfolio, np.hstack((incs, deq)))
                else:
                    incs = incs
                if space <= 0:
                    for _ in range(space, 0):
                        rm = self.portfolio.pop(0)
                        self.logger.info('Removing configuration from portfolio:\n%s', rm)
                        self.dequeued.append(rm)
            else:
                incs = incs[to_keep_ids][:self.incs_per_round]

            read(self.rh, os.path.join(self.top_dir, 'psmac3*', 'run_*'), self.scenario.cs, self.logger)
            self.logger.info('Kept incumbents')
            fit = True
            for inc in incs:
                self.logger.info(inc)
                self.candidate_configs_cost_per_inst[inc] = self.predict_missing_data(cost_per_conf[inc], inc, fit)
                fit = False
            cur_portfolio_cost = self._update_portfolio(incs, self.candidate_configs_cost_per_inst)
            if portfolio_cost <= cur_portfolio_cost:
                self.logger.info("No further progress (%f) --- terminate hydra", portfolio_cost)
                break
            else:
                portfolio_cost = cur_portfolio_cost
                self.logger.info("Current pertfolio cost: %f", portfolio_cost)

            # modify TAE such that it return oracle performance
            self.tae = ExecuteTARunHydra(ta=self.scenario.ta, run_obj=self.scenario.run_obj,
                                         cost_oracle=self.cost_per_inst, tae=self._tae)

            self.scenario.output_dir = os.path.join(self.top_dir, "psmac3-output_%s" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')))
            self.output_dir = create_output_directory(self.scenario, run_id=self.run_id, logger=self.logger)
        self.rh.save_json(fn=os.path.join(self.top_dir, 'all_validated_runs_runhistory.json'), save_external=True)
        while self.dequeued and len(self.portfolio) < self.max_size:
            self.portfolio.append(self.dequeued.pop(0))
        with open(os.path.join(self.top_dir, 'portfolio.pkl'), 'wb') as fh:
            pickle.dump(self.portfolio, fh)
        self.logger.info("~"*120)
        self.logger.info('Resulting Portfolio:')
        for configuration in self.portfolio:
            self.logger.info(str(configuration))
        self.logger.info("~"*120)

        return self.portfolio

    def _update_portfolio(self, incs: np.ndarray, config_cost_per_inst: typing.Dict) -> typing.Union[np.float, float]:
        """
        Validates all configurations (in incs) and determines which ones to add to the portfolio

        Parameters
        ----------
        incs: np.ndarray
            List of Configurations

        Returns
        -------
        cur_cost: typing.Union[np.float, float]
            The current cost of the portfolio

        """
        self.cost_per_inst = None
        for kept in incs:
            if kept not in self.portfolio:
                self.portfolio.append(kept)
        # we have to recompute cost_per_inst for every new portfolio as we sometimes can throw out old configurations
        for kept in self.portfolio:
            cost_per_inst = config_cost_per_inst[kept]
            if self.cost_per_inst:
                for key in cost_per_inst:
                    self.cost_per_inst[key] = min(self.cost_per_inst[key], cost_per_inst[key])
            else:
                self.cost_per_inst = cost_per_inst
        cur_cost = np.mean(list(self.cost_per_inst.values()))  # type: float
        return cur_cost
