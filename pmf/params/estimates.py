from pmf.params.quantities import InferenceQuantities


class ParamsEstimates:

    def __init__(self, inference_quantities: InferenceQuantities):
        self.inference_quantities = inference_quantities

    @property
    def x(self):
        return self.inference_quantities.eg_x

    @property
    def y(self):
        return self.inference_quantities.eg_y

    @property
    def rhox(self):
        return self.inference_quantities.etg_rhox

    @property
    def rhoy(self):
        return self.inference_quantities.etg_rhoy
