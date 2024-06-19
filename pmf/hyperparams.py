class ModelHyperParams:
    """
    The hyperparameters for the initial model.
    """

    def __init__(
        self,
        num_features: int,
        a_x,
        a_y,
        b_x,
        b_y,
        c_x,
        c_y,
        alpha_x,
        alpha_y,
        beta_x,
        beta_y,
    ):
        """
        Store the model hyperparameters.
        """

        assert num_features > 0

        assert a_x > 0
        assert a_y > 0
        assert b_x > 0
        assert b_y > 0
        assert c_x > 0
        assert c_y > 0
        assert alpha_x > 0
        assert alpha_y > 0
        assert beta_x > 0
        assert beta_y > 0

        self.num_features = num_features

        self.a_x = a_x
        self.a_y = a_y
        self.b_x = b_x
        self.b_y = b_y
        self.c_x = c_x
        self.c_y = c_y
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.beta_x = beta_x
        self.beta_y = beta_y
