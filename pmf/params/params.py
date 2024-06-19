from enum import Enum, auto
import numpy as np
import sparse

from pmf.hyperparams import ModelHyperParams


class ChangedParameter(Enum):
    ZN = auto()
    X = auto()
    Y = auto()
    ZETAX = auto()
    ZETAY = auto()
    RHOX = auto()
    RHOY = auto()


class Params:
    """
    Parameters for the Variational Inference Model.

    Parameters
    ----------
        ZN: PoissonPlus-Multinomial distribution
    :param zn_poisson_rate: sparse.COO of shape (T, N1, N2)
    :param zn_multinomial_prob: sparse.COO of shape (T, N1, N2, d)

        X and Y: Gamma distributions
    :param x_shape: np.ndarray of shape (N1, d)
    :param x_rate: np.ndarray of shape (N1, d)
    :param y_shape: np.ndarray of shape (N2, d)
    :param y_rate: np.ndarray of shape (N2, d)

        ZetaX and ZetaY: Gamma distributions
    :param zetax_shape: scalar equal to d * a_x + b_x
    :param zetax_rate: np.ndarray of shape (d,)
    :param zetay_shape: scalar equal to d * a_y + b_y
    :param zetay_rate: np.ndarray of shape (d,)

        RhoX and RhoY: Truncated Gamma distributions
    :param rhox_shape: np.ndarray of shape (T, N1)
    :param rhox_rate: np.ndarray of shape (T, N1)
    :param rhoy_shape: np.ndarray of shape (T, N2)
    :param rhoy_rate: np.ndarray of shape (T, N2)
    ----------
    """

    def __init__(self, hyperparams: ModelHyperParams, time: int, N1: int, N2: int):
        self._time = time
        self._N1 = N1
        self._N2 = N2
        self._zn_poisson_rate = None
        self._zn_multinomial_prob = None
        self._x_shape = None
        self._x_rate = None
        self._y_shape = None
        self._y_rate = None
        self._zetax_shape = hyperparams.num_features * hyperparams.a_x + hyperparams.b_x
        self._zetax_rate = None
        self._zetay_shape = hyperparams.num_features * hyperparams.a_y + hyperparams.b_y
        self._zetay_rate = None
        self._rhox_shape = None
        self._rhox_rate = None
        self._rhoy_shape = None
        self._rhoy_rate = None

        self._eval_needed = False

    def set_eval_needed(self, eval_needed: bool):
        self._eval_needed = eval_needed

    def register_inference_quantities(self, inference_quantities):
        self.inference_quantities = inference_quantities

    def register_evaluation_quantities(self, evaluation_quantities):
        self.evaluation_quantities = evaluation_quantities

    def _notify_change(self, changed: ChangedParameter):
        self.inference_quantities.update(changed)
        if self._eval_needed:
            self.evaluation_quantities.update(changed)

    def set_zn(self, zn_poisson_rate: sparse.COO, zn_multinomial_prob: sparse.COO):
        self._zn_poisson_rate = zn_poisson_rate
        self._zn_multinomial_prob = zn_multinomial_prob
        self._notify_change(ChangedParameter.ZN)

    def set_x(self, x_shape: np.ndarray, x_rate: np.ndarray):
        self._x_shape = x_shape
        self._x_rate = x_rate
        self._notify_change(ChangedParameter.X)

    def set_y(self, y_shape: np.ndarray, y_rate: np.ndarray):
        self._y_shape = y_shape
        self._y_rate = y_rate
        self._notify_change(ChangedParameter.Y)

    def set_zetax_rate(self, zetax_rate: np.ndarray):
        self._zetax_rate = zetax_rate
        self._notify_change(ChangedParameter.ZETAX)

    def set_zetay_rate(self, zetay_rate: np.ndarray):
        self._zetay_rate = zetay_rate
        self._notify_change(ChangedParameter.ZETAY)

    def set_rhox(self, rhox_shape: np.ndarray, rhox_rate: np.ndarray):
        self._rhox_shape = rhox_shape
        self._rhox_rate = rhox_rate
        self._notify_change(ChangedParameter.RHOX)

    def set_rhoy(self, rhoy_shape: np.ndarray, rhoy_rate: np.ndarray):
        self._rhoy_shape = rhoy_shape
        self._rhoy_rate = rhoy_rate
        self._notify_change(ChangedParameter.RHOY)

    @property
    def time(self):
        return self._time

    @property
    def N1(self):
        return self._N1

    @property
    def N2(self):
        return self._N2

    @property
    def zn_poisson_rate(self):
        return self._zn_poisson_rate

    @property
    def zn_multinomial_prob(self):
        return self._zn_multinomial_prob

    @property
    def x_shape(self):
        return self._x_shape

    @property
    def x_rate(self):
        return self._x_rate

    @property
    def y_shape(self):
        return self._y_shape

    @property
    def y_rate(self):
        return self._y_rate

    @property
    def zetax_shape(self):
        return self._zetax_shape

    @property
    def zetax_rate(self):
        return self._zetax_rate

    @property
    def zetay_shape(self):
        return self._zetay_shape

    @property
    def zetay_rate(self):
        return self._zetay_rate

    @property
    def rhox_shape(self):
        return self._rhox_shape

    @property
    def rhox_rate(self):
        return self._rhox_rate

    @property
    def rhoy_shape(self):
        return self._rhoy_shape

    @property
    def rhoy_rate(self):
        return self._rhoy_rate
