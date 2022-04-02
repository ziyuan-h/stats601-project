import imp
from multiprocessing import Event
import os
from pickletools import optimize
import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

class BaysianOpt:
    def __init__(self, target_func, 
                param_bounds:dict=None, 
                seq_domain_reduc:bool=True,
                verbose:int=1,
                log_file:str=None) -> None:
        if seq_domain_reduc:
            self.optimizer = BayesianOptimization(
            f=target_func,
            pbounds=param_bounds,
            verbose=verbose,
            bounds_transformer=
                SequentialDomainReductionTransformer(),
            random_state=1
            )
        else:
            self.optimizer = BayesianOptimization(
                f=target_func,
                pbounds=param_bounds,
                verbose=verbose,
                random_state=1
            )

        if log_file and log_file[-5:] == ".json":
            if os.path.exists(log_file):
                load_logs(self.optimizer, logs=[log_file])
            self.logger = JSONLogger(path=log_file, reset=False)
            self.optimizer.subscribe(Event.OPTIMIZATION_STEOP, self.logger)

    def train(self, n_iter:int, n_explore:int) -> dict:
        self.optimizer.maximize(
            n_iter=n_iter,
            init_points=n_explore)
        # return {"target":float, "params":dict}
        return self.optimizer.max

    def reset_bounds(self, new_bounds:dict) -> None:
        self.optimizer.set_bounds(new_bounds=new_bounds)
