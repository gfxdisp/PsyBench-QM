from pyfvvdp import fvvdp

class fvvdp_met(fvvdp):

    @staticmethod
    def name():
        return 'FovVideoVDP'

    @staticmethod
    def is_lower_better():
        return False