from abc import ABC, abstractmethod
import numpy as np
import sparse

from pmf.params.params import ChangedParameter, Params
from utils.distributions import loggamma, logtruncatedgamma, poissonplus, truncatedgamma


class Quantities(ABC):

    @abstractmethod
    def update(self, changed: ChangedParameter):
        pass


class InferenceQuantities(Quantities):
    """
    Quantities related to the parameters of the Variational Inference Model
    that are used at every iteration.

    Notation
    --------
    EG: Expectation of the Gamma distribution
    ELG: Expectation of the Log-Gamma distribution
    ETG: Expectation of the Truncated Gamma distribution
    ELTG: Expectation of the Log-Truncated Gamma distribution

    Quantities
    ----------
        ZN: PoissonPlus-Multinomial distribution
    zn_poisson_plus_rate: sparse.COO of shape (T, N1, N2)
    zn_prod_poisson_plus_rate_multinomial_prob: sparse.COO of shape (T, N1, N2, d)

        X and Y: Gamma distributions
    eg_x: np.ndarray of shape (N1, d)
    eg_y: np.ndarray of shape (N2, d)
    elg_x: np.ndarray of shape (N1, d)
    elg_y: np.ndarray of shape (N2, d)

        ZetaX and ZetaY: Gamma distributions
    eg_zetax: np.ndarray of shape (d,)
    eg_zetay: np.ndarray of shape (d,)

        RhoX and RhoY: Truncated Gamma distributions
    etg_rhox: np.ndarray of shape (T, N1)
    eltg_rhox: np.ndarray of shape (T, N1)
    etg_rhoy: np.ndarray of shape (T, N2)
    eltg_rhoy: np.ndarray of shape (T, N2)
    """

    def __init__(self, params: Params):
        self.params = params
        # Register the observer
        self.params.register_inference_quantities(self)

        self._zn_poisson_plus_rate = None
        self._zn_prod_poisson_plus_rate_multinomial_prob = None

        self._eg_x = None
        self._eg_y = None
        self._elg_x = None
        self._elg_y = None

        self._eg_zetax = None
        self._eg_zetay = None

        self._etg_rhox = None
        self._eltg_rhox = None
        self._etg_rhoy = None
        self._eltg_rhoy = None

    def update(self, changed: ChangedParameter):
        match changed:
            case ChangedParameter.ZN:
                self._zn_poisson_plus_rate = poissonplus.rate(
                    self.params.zn_poisson_rate
                )
                self._zn_prod_poisson_plus_rate_multinomial_prob = (
                    self._compute_prod_poisson_plus_rate_multinomial_prob()
                )

            case ChangedParameter.X:
                self._eg_x = self.params.x_shape / self.params.x_rate
                self._elg_x = loggamma.expectation(
                    self.params.x_shape, self.params.x_rate
                )
            case ChangedParameter.Y:
                self._eg_y = self.params.y_shape / self.params.y_rate
                self._elg_y = loggamma.expectation(
                    self.params.y_shape, self.params.y_rate
                )

            case ChangedParameter.ZETAX:
                self._eg_zetax = self.params.zetax_shape / self.params.zetax_rate
            case ChangedParameter.ZETAY:
                self._eg_zetay = self.params.zetay_shape / self.params.zetay_rate

            case ChangedParameter.RHOX:
                self._etg_rhox = truncatedgamma.expectation(
                    self.params.rhox_shape, self.params.rhox_rate
                )
                self._eltg_rhox = logtruncatedgamma.expectation(
                    self.params.rhox_shape, self.params.rhox_rate
                )
            case ChangedParameter.RHOY:
                self._etg_rhoy = truncatedgamma.expectation(
                    self.params.rhoy_shape, self.params.rhoy_rate
                )
                self._eltg_rhoy = logtruncatedgamma.expectation(
                    self.params.rhoy_shape, self.params.rhoy_rate
                )

            case _:
                raise ValueError(f"Invalid parameter: {changed}")

    def _compute_prod_poisson_plus_rate_multinomial_prob(self) -> sparse.COO:
        # Transpose the matrix in order to use the broadcasting mechanism of numpy,
        # which is more efficient than adding a new axis to the matrix
        transposed_zn_multinomial_prob = np.transpose(
            self.params.zn_multinomial_prob, (3, 0, 1, 2)
        )
        prod = transposed_zn_multinomial_prob * self._zn_poisson_plus_rate
        return np.transpose(prod, (1, 2, 3, 0))

    @property
    def zn_poisson_plus_rate(self):
        return self._zn_poisson_plus_rate

    @property
    def zn_prod_poisson_plus_rate_multinomial_prob(self):
        return self._zn_prod_poisson_plus_rate_multinomial_prob

    @property
    def eg_x(self):
        return self._eg_x

    @property
    def eg_y(self):
        return self._eg_y

    @property
    def elg_x(self):
        return self._elg_x

    @property
    def elg_y(self):
        return self._elg_y

    @property
    def eg_zetax(self):
        return self._eg_zetax

    @property
    def eg_zetay(self):
        return self._eg_zetay

    @property
    def etg_rhox(self):
        return self._etg_rhox

    @property
    def etg_rhoy(self):
        return self._etg_rhoy

    @property
    def eltg_rhox(self):
        return self._eltg_rhox

    @property
    def eltg_rhoy(self):
        return self._eltg_rhoy


class EvaluationQuantities(Quantities):
    """
    Quantities related to the parameters of the Variational Inference Model
    that are used when the model needs to be evaluated.

    Notation
    --------
    EG: Expectation of the Gamma distribution
    ELG: Expectation of the Log-Gamma distribution
    ETG: Expectation of the Truncated Gamma distribution
    ELTG: Expectation of the Log-Truncated Gamma distribution

    Quantities
    ----------
        ZetaX and ZetaY: Gamma distributions
    elg_zetax: np.ndarray of shape (d,)
    elg_zetay: np.ndarray of shape (d,)
    """

    def __init__(self, params: Params):
        self.params = params
        # Register the observer
        self.params.register_evaluation_quantities(self)

        self._elg_zetax = None
        self._elg_zetay = None

    def update(self, changed: ChangedParameter):
        match changed:
            case ChangedParameter.ZN:
                return

            case ChangedParameter.X:
                return

            case ChangedParameter.Y:
                return

            case ChangedParameter.ZETAX:
                self._elg_zetax = loggamma.expectation(
                    self.params.zetax_shape, self.params.zetax_rate
                )
            case ChangedParameter.ZETAY:
                self._elg_zetay = loggamma.expectation(
                    self.params.zetay_shape, self.params.zetay_rate
                )

            case ChangedParameter.RHOX:
                return
            case ChangedParameter.RHOY:
                return

            case _:
                raise ValueError(f"Invalid parameter: {changed}")

    @property
    def elg_zetax(self):
        return self._elg_zetax

    @property
    def elg_zetay(self):
        return self._elg_zetay
