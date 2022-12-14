"""

"""
from numpy import array


class TowerCostModel:

    def __init__(self, sf_cost: float, tw_cost: float, re_cost: float, tes_cost: float, pb: float):

        self.costs = array([sf_cost, tw_cost, re_cost, tes_cost, pb])


class TowerPowerPlant:

    def __init__(self, sf: float, tw: float, re: float, tes: float, pb: float):

        self.solar_field = sf
        self.tower = tw
        self.receiver = re
        self.tes = tes
        self.power_block = pb

    def capex(self, cost_model: TowerCostModel):
        pass

    def opex(self, cost_model):
        pass


class ParabolicPowerPlant:

    def __init__(self, sf, tes, pb):

        self.solar_field = sf
        self.tes = tes
        self.power_block = pb

    def capex(self, cost_model):
        pass

    def opex(self, cost_model):
        pass
