#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
import os
import inspect
import glob
import pickle

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_askl_surrogate_run import ExecuteASKLRun
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.constants import MAXINT
from smac.utils.validate import Validator

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    req_opts = parser.add_argument_group("Required Options")
    req_opts.add_argument("--scenario", required=True,
                          help="path to SMAC scenario")
    req_opts.add_argument("--dir", required=True,
                          help="Path to Hydra output containing all pSMAC runs")

    req_opts = parser.add_argument_group("Optional Options")
    req_opts.add_argument("--configs", default="all", type=str,
                          choices=["final", "all"],
                          help="what configurations to evaluate. "
                               "final=final portfolio; all=trjaectory of portfolios")
    req_opts.add_argument("--instances", default="train", type=str,
                          choices=["train", "test", "train+test"],
                          help="what instances to evaluate")
    req_opts.add_argument('--epm', dest='epm', action='store_true',
                          help="Use EPM to validate")
    req_opts.add_argument('--no-epm', dest='epm', action='store_false',
                          help="Don't use EPM to validate")
    req_opts.set_defaults(epm=False)
    req_opts.add_argument("--runhistory", default=None, type=str, nargs='*',
                          help="path to one or more runhistories to take runs "
                               "from to either avoid recalculation or to train"
                               " the epm")
    req_opts.add_argument("--seed", type=int, help="random seed")
    req_opts.add_argument("--repetitions", default=1, type=int,
                          help="number of repetitions for nondeterministic "
                               "algorithms")
    req_opts.add_argument("--n_jobs", default=1, type=int,
                          help="number of cpu-cores to use (-1 to use all)")
    req_opts.add_argument("--tae", default="old", type=str,
                          help="what tae to use (if not using epm)", choices=["aclib", "old", "askl"])
    req_opts.add_argument("--verbose_level", default="INFO",
                          choices=["INFO", "DEBUG"],
                          help="verbose level")

    args_, misc = parser.parse_known_args()

    # remove leading '-' in option names
    misc = dict((k.lstrip("-"), v.strip("'"))
                for k, v in zip(misc[::2], misc[1::2]))

    if args_.verbose_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    scenario = Scenario(args_.scenario, cmd_options={'output_dir': ""})
    # traj_logger = TrajLogger(None, Stats(scenario))
    # trajectory = traj_logger.read_traj_aclib_format(args_.trajectory, scenario.cs)
    if args_.tae == "old":
        tae = ExecuteTARunOld(ta=scenario.ta,
                              run_obj=scenario.run_obj,
                              par_factor=scenario.par_factor,
                              cost_for_crash=scenario.cost_for_crash)
    if args_.tae == "aclib":
        tae = ExecuteTARunAClib(ta=scenario.ta,
                                run_obj=scenario.run_obj,
                                par_factor=scenario.par_factor,
                                cost_for_crash=scenario.cost_for_crash)
    if args_.tae == 'askl':
        tae = ExecuteASKLRun(ta=[''],
                             run_obj=scenario.run_obj,
                             par_factor=scenario.par_factor,
                             cost_for_crash=scenario.cost_for_crash)

    all_configs = []
    load_from = list(sorted(glob.glob(os.path.join(args_.dir, 'psmac*', 'portfolio*'))))
    # load_from.extend(list(glob.glob(os.path.join(args_.dir, 'portfolio*'))))
    for portfoliofn in load_from:
        with open(portfoliofn, 'rb') as fh:
            portfolio = pickle.load(fh)

        validator = Validator(scenario, None, args_.seed)
        out_run_dir = list(glob.glob(os.path.join(os.path.dirname(portfoliofn), '*%d' % MAXINT)))
        assert len(out_run_dir) <= 1
        if len(out_run_dir):
            out_run_dir = out_run_dir[0]
            os.rename(os.path.join(out_run_dir, 'validated_runhistory.json'), os.path.join(out_run_dir, 'old_vrh.json'))

        # Load runhistory
        if args_.runhistory:
            runhistory = RunHistory(average_cost)
            for rh_path in args_.runhistory:
                runhistory.update_from_json(rh_path, scenario.cs)
        else:
            runhistory = None

        if args_.epm:
            validator.validate_epm(config_mode=portfolio,
                                   instance_mode=args_.instances,
                                   repetitions=args_.repetitions,
                                   runhistory=runhistory,
                                   output_fn=os.path.join(out_run_dir, 'validated_runhistory.json'))
        else:
            validator.validate(config_mode=portfolio,
                               instance_mode=args_.instances,
                               repetitions=args_.repetitions,
                               n_jobs=args_.n_jobs,
                               runhistory=runhistory,
                               tae=tae, output_fn=os.path.join(out_run_dir, 'validated_runhistory.json'))
