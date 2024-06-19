from pmf.params.params import Params
from pmf.params.updates import Updates


def init_zeta(
    params: Params,
    updates: Updates,
):
    """
    Initialize the parameters for the zeta components.
    """
    # Initialize the parameters for the zeta-x component.
    zetax_rate = updates.zetax()
    params.set_zetax_rate(zetax_rate)

    # Initialize the parameters for the zeta-y component.
    zetay_rate = updates.zetay()
    params.set_zetay_rate(zetay_rate)
