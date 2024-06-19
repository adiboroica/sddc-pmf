from typing import Tuple
import sparse

from pmf.hyperparams import ModelHyperParams
from pmf.params.params import Params
from pmf.params.quantities import EvaluationQuantities, InferenceQuantities
from pmf.params.updates import Updates
from pmf.params.initializations.init_rho import init_rho
from pmf.params.initializations.init_xy import (
    InitXYStrategy,
    InitXYsvd,
    InitXYconstant,
    InitXYrandomBeta,
    InitXYrandomUniform,
    InitXYrandomGamma,
)
from pmf.params.initializations.init_zeta import init_zeta
from pmf.params.initializations.init_zn import init_zn
from utils.logger import Logger


class InitParams:

    def __init__(
        self,
        xy_strategy: InitXYStrategy = InitXYsvd(),
    ):
        self.xy_strategy = xy_strategy


def initialize_params(
    init_params: InitParams,
    data: sparse.COO,
    hyperparams: ModelHyperParams,
    logger: Logger,
) -> Tuple[Params, InferenceQuantities, EvaluationQuantities, Updates]:
    # Extract necessary variables
    T, N1, N2 = data.shape
    xy_strategy = init_params.xy_strategy

    # Initialize the objects to hold the parameters, quantities, and updates.
    params = Params(hyperparams, T, N1, N2)
    inference_quantities = InferenceQuantities(params)
    evaluation_quantities = EvaluationQuantities(params)
    updates = Updates(hyperparams, params, inference_quantities)

    # Initialize the parameters for the Z, N components.
    init_zn(data, hyperparams, params)
    logger.info("Initialized: Z, N")

    # Initialize the parameters for the X, Y components.
    if isinstance(xy_strategy, InitXYsvd):
        xy_strategy.init(data, hyperparams, params)
    elif isinstance(xy_strategy, InitXYconstant):
        xy_strategy.init(hyperparams, params)
    elif isinstance(xy_strategy, InitXYrandomUniform):
        xy_strategy.init(hyperparams, params)
    elif isinstance(xy_strategy, InitXYrandomBeta):
        xy_strategy.init(hyperparams, params)
    elif isinstance(xy_strategy, InitXYrandomGamma):
        xy_strategy.init(hyperparams, params)
    else:
        raise NotImplementedError(f"Unknown strategy: {xy_strategy}")
    logger.info("Initialized: X, Y")

    # Initialize the parameters for the Zeta components.
    init_zeta(params, updates)
    logger.info("Initialized: Zeta")

    # Initialize the parameters for the Rho components.
    init_rho(data, params, inference_quantities, updates)
    logger.info("Initialized: Rho")

    return params, inference_quantities, evaluation_quantities, updates
