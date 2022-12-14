

import subprocess
from pathlib import Path
from numpy import zeros, power, array
from scipy.optimize import fsolve

#######################################################################################################################
#######################################################################################################################


class PV_CostModel:

    def __init__(self, module: float, inverter: float, bos: float, install: float, soft: float, OeM: float):
        self.module = module  # in USD/kW-DC
        self.inverter = inverter  # in USD/kW-AC
        self.bos = bos  # in USD/kW-DC
        self.install = install  # in USD/kW-DC

        self.soft = soft  # in USD/kW-DC
        # Soft costs include: Margin, Financing costs, System design, Permitting, and Incentive application.
        # It is very close to EPC indirect costs which compose the CSP cost model.

        self.OeM = OeM  # in USD/kW-DC


class TS_CostModel:

    def __init__(self, site: float, sf: float, tes: float, bop: float, pb: float, aux_heater: float, services: float,
                 margin_contingencies: float, development: float, owners: float, land=1.0, infrastructure=6.e6):

        self.site_ec = abs(site)
        self.sf_ec = abs(sf)
        self.tes_ec = abs(tes)
        self.bop_ec = abs(bop)
        self.pb_ec = abs(pb)
        self.heater_ec = abs(aux_heater)

        self.land = abs(land)
        self.infrastructure = abs(infrastructure)

        if services >= 1.0:
            raise "EPC Services cost should be between 0 and 1. It is a fraction of the total EPC Direct Costs"
        else:
            self.services = abs(services)

        if margin_contingencies >= 1.0:
            raise "Margin and Contingencies costs should be between 0 and 1. " \
                  "It is a fraction of the total EPC Direct Costs"
        else:
            self.margin = abs(margin_contingencies)

        if development >= 1.0:
            raise "Project Development cost should be between 0 and 1. It is a fraction of the total EPC Direct Costs"
        else:
            self.development = abs(development)

        if owners >= 1.0:
            raise "Additional Owner's cost should be between 0 and 1. It is a fraction of the total EPC Direct Costs"
        else:
            self.owners = abs(owners)


class PV_module:

    def __init__(self, rating: float, area: float, eff: float, inv_eff: float, inv_losses: float,
                 soling_loss: float, dc_loss: float, deg_rate: float):
        self.peak_eff = abs(eff)
        self.module_rating = abs(rating)
        self.area = abs(area)

        self.area_power_ratio = (self.module_rating / 1000) / self.area

        self.soiling = abs(soling_loss)
        self.dc_loss = abs(dc_loss)
        self.deg_rate = abs(deg_rate)

        self.inv_eff = abs(inv_eff)
        self.inv_losses = abs(inv_losses)


class PT_Module:

    def __init__(self, opt_efficiency: float, module_area: float):

        self.eta_zero = opt_efficiency
        self.module_area = module_area


class TS:

    def __init__(self, flh: float, turbine_gross: float, turbine_net: float, pb_eff: float, heater_power: float = None):
        """
        A Class to defined the Thermal Systems (TS) of a PV-P2H-TS power plant
        :param flh: Full Load Hours (flh) of base load production of the power block
        :param turbine_gross: Steam turbine base load gross output, in MW-e
        :param turbine_net: Steam turbine base load net output, in MW-e
        :param pb_eff: Power block gross efficiency
        :param heater_power: Auxiliary heater power, in MW-th
        """

        # number of full load hours of power block base load production stored at the TES
        self.flh = abs(flh)  # in hours

        self.pb_gross_output = 1000 * abs(turbine_gross)  # in kW-e
        self.pb_net_output = 1000 * abs(turbine_net)  # in kW-e
        self.heater_power = 0.0 if heater_power is None else 1000 * abs(heater_power)  # in kW-th

        # Power block gross efficiency.
        # Basically the steam cycle efficiency since the TES has an almost 100% efficiency. [Gordon et al., 2021]
        self.pb_eff = abs(pb_eff)

        # stored energy at the TES system, in kWh-th.
        self.tes_energy = self.flh * (self.pb_gross_output / self.pb_eff)


class PTPP:

    def __init__(self, sf_area: float, ts: TS, ts_cost: TS_CostModel):
        #####################################################
        # Solar field and land area data
        self.sf_area = abs(sf_area)
        self.land_area = self.sf_area * 4.0
        ######################################################

        ######################################################
        # Thermal systems data
        self.flh = ts.flh  # number of hours of full (base) load hours of operation -- related
        self.tes_energy = ts.tes_energy  # stored energy which correspond to the nbr of flh, in kWh
        self.heater_power = ts.heater_power  # auxiliary heater power in kW-e
        self.pb_gross_output = ts.pb_gross_output  # power block gross output, in kW-e
        self.pb_net_output = ts.pb_net_output  # power block net output, in kW-e
        ######################################################

        ######################################################
        # Thermal Systems cost calculation, all in USD
        # Breakdown of EPC direct costs
        self.site_cost = self.sf_area * ts_cost.site_ec
        self.sf_cost = self.sf_area * ts_cost.sf_ec
        self.tes_cost = ts.tes_energy * ts_cost.tes_ec
        self.bop_cost = ts.pb_gross_output * ts_cost.bop_ec
        self.pb_cost = ts.pb_gross_output * ts_cost.pb_ec
        ######################################################

        ######################################################
        # EPC Direct Cost, all in USD
        self.ts_epc_direct_cost = (self.sf_cost + self.site_cost + self.tes_cost + self.bop_cost + self.pb_cost)
        ######################################################

        ######################################################
        #  EPC Indirect cost, all in USD
        self.ts_epc_indirect_cost = self.ts_epc_direct_cost * (ts_cost.services + ts_cost.margin)
        ######################################################

        ######################################################
        # Total EPC Cost, all in USD
        self.ts_epc_cost = self.ts_epc_direct_cost + self.ts_epc_indirect_cost
        ######################################################

        ######################################################
        # Owner's cost, all in USD
        self.land_cost = self.land_area * ts_cost.land
        self.ts_owners_cost = self.ts_epc_cost * (ts_cost.development + ts_cost.owners)
        self.ts_owners_cost += self.land_cost + ts_cost.infrastructure
        ######################################################

        ######################################################
        # CAPEX and OPEX calculations, all in USD
        self.capex = self.ts_epc_cost + self.ts_owners_cost
        self.opex = self.capex * (2.2 / 100)

    def economic_assessment(self, sf2grid: float, tes2grid: float, pv_degradation: float, ts_degradation: float,
                            discount_rate: float, inflation_rate: float, n=20):

        # 08.april.2022 ################################################################################################
        # Pedro's version of investment analysis, as shown in the Google Sheet file.
        # This Python implementation should follow the sheet equations.
        ################################################################################################################

        if pv_degradation >= 1 or ts_degradation >= 1:
            raise "Degradation rate should be between 0 and 1."
        else:
            pv_deg_rate, ts_deg_rate = abs(pv_degradation), abs(ts_degradation)

        # energy flow
        pv2grid_energy_flow = zeros(n + 1)
        pv2grid_energy_flow[1:] = [sf2grid * power(1 - pv_deg_rate, i) for i in range(1, n + 1)]

        tes2grid_energy_flow = zeros(n + 1)
        tes2grid_energy_flow[1:] = [tes2grid * power(1 - ts_deg_rate, i) for i in range(1, n + 1)]

        if discount_rate >= 1 or inflation_rate >= 1:
            raise "Discount and Inflation rate should be between 0 and 1"
        else:
            rd, ri = discount_rate, inflation_rate

        discount_flow = zeros(n + 1)
        discount_flow[:] = [power(1 + rd, -i) for i in range(n + 1)]

        # cost_flow
        cost_flow = zeros(n + 1)
        cost_flow[0] = -self.capex
        cost_flow[1:] = [-self.opex * power(1 + ri, i) for i in range(1, n + 1)]

        energy_flow = pv2grid_energy_flow + tes2grid_energy_flow

        def yield_cash_flow(x: float): return energy_flow * x
        def total_cash_flow(x: float): return yield_cash_flow(x) + cost_flow
        def npv(x: float): return total_cash_flow(x).dot(discount_flow)

        x0 = self.capex / (n * (tes2grid + sf2grid))

        lcoe = fsolve(npv, array([x0]))[0]

        return lcoe

    def capacity_factor(self, sf2grid: float, tes2grid: float):
        return (sf2grid + tes2grid) / (8.760 * self.pb_net_output)


class PVTS:

    def __init__(self, pv_modules: int, pv_module: PV_module, pv_cost: PV_CostModel, ts: TS, ts_cost: TS_CostModel):

        ######################################################
        # Photovoltaic field data
        self.pv_area = abs(pv_modules) * pv_module.area
        self.land_area = self.pv_area * 2.25

        self.pv_power_dc = self.pv_area * pv_module.area_power_ratio  # in kW-DC
        self.pv_power_ac = self.pv_power_dc / (pv_module.inv_eff * (1 - pv_module.inv_losses))  # in kW-AC
        ######################################################

        ######################################################
        # Thermal systems data
        self.flh = ts.flh  # number of hours of full (base) load hours of operation -- related
        self.tes_energy = ts.tes_energy  # stored energy which correspond to the nbr of flh, in kWh
        self.heater_power = ts.heater_power  # auxiliary heater power in kW-e
        self.pb_gross_output = ts.pb_gross_output  # power block gross output, in kW-e
        self.pb_net_output = ts.pb_net_output  # power block net output, in kW-e
        ######################################################

        ######################################################
        # Photovoltaic field cost calculations, all in USD
        self.pv_cost = (pv_cost.module + pv_cost.bos + pv_cost.install) * self.pv_power_dc  # in USD
        self.inverters_cost = self.pv_power_ac * pv_cost.inverter  # in USD

        self.pv_epc_direct_cost = self.pv_cost + self.inverters_cost  # in USD
        self.pv_epc_indirect_cost = self.pv_power_dc * pv_cost.soft  # in USD
        # Photovoltaic field Operation and Maintenance (O&M) costs, in USD
        self.pv_om = self.pv_power_dc * pv_cost.OeM  # in USD
        #######################################################

        #######################################################
        # Thermal Systems EPC Direct Costs calculation, all in USD
        self.site_cost = self.land_area * (ts_cost.site_ec / 4.0)
        self.tes_cost = ts.tes_energy * ts_cost.tes_ec
        self.bop_cost = ts.pb_gross_output * ts_cost.bop_ec
        self.pb_cost = ts.pb_gross_output * ts_cost.pb_ec
        self.heater_cost = ts.heater_power * ts_cost.heater_ec
        self.ts_epc_direct_cost = (self.site_cost + self.tes_cost + self.bop_cost + self.pb_cost + self.heater_cost)
        #######################################################

        #######################################################
        # Thermal Systems EPC Indirect Costs calculation, all in USD
        self.ts_epc_indirect_cost = self.ts_epc_direct_cost * (ts_cost.services + ts_cost.margin)
        #######################################################

        #######################################################
        # Thermal Systems Total EPC Costs calculation, all in USD
        self.ts_epc_cost = self.ts_epc_direct_cost + self.ts_epc_indirect_cost
        #######################################################

        #######################################################
        # Thermal Systems Owner's Costs calculation, all in USD
        self.land_cost = self.land_area * ts_cost.land
        self.ts_owners_cost = self.ts_epc_cost * (ts_cost.development + ts_cost.owners)
        self.ts_owners_cost += self.land_cost + ts_cost.infrastructure
        #######################################################

        #######################################################
        # Power Plant CAPEX and OPEX calculations, in USD
        self.capex = self.pv_epc_direct_cost + self.pv_epc_indirect_cost + self.ts_epc_cost + self.ts_owners_cost
        self.opex = self.pv_om + (2.2 / 100) * (self.ts_epc_cost + self.ts_owners_cost)
        #######################################################

    def economic_assessment(self, pv2grid: float, tes2grid: float, pv_degradation: float, ts_degradation: float,
                            discount_rate: float, inflation_rate: float, n=20):

        # 08.april.2022 ################################################################################################
        # Pedro's version of investment analysis, as shown in the Google Sheet file.
        # This Python implementation should follow the sheet equations.
        ################################################################################################################

        if pv_degradation >= 1 or ts_degradation >= 1:
            raise "Degradation rate should be between 0 and 1."
        else:
            pv_deg_rate, ts_deg_rate = abs(pv_degradation), abs(ts_degradation)

        # energy flow
        pv2grid_energy_flow = zeros(n + 1)
        pv2grid_energy_flow[1:] = [pv2grid * power(1 - pv_deg_rate, i) for i in range(1, n + 1)]

        tes2grid_energy_flow = zeros(n + 1)
        tes2grid_energy_flow[1:] = [tes2grid * power(1 - ts_deg_rate, i) for i in range(1, n + 1)]

        if discount_rate >= 1 or inflation_rate >= 1:
            raise "Discount and Inflation rate should be between 0 and 1"
        else:
            rd, ri = discount_rate, inflation_rate

        discount_flow = zeros(n + 1)
        discount_flow[:] = [power(1 + rd, -i) for i in range(n + 1)]

        # cost_flow
        cost_flow = zeros(n + 1)
        cost_flow[0] = -self.capex
        cost_flow[1:] = [-self.opex * power(1 + ri, i) for i in range(1, n + 1)]

        energy_flow = pv2grid_energy_flow + tes2grid_energy_flow

        def yield_cash_flow(x: float): return energy_flow * x
        def total_cash_flow(x: float): return yield_cash_flow(x) + cost_flow
        def npv(x: float): return total_cash_flow(x).dot(discount_flow)

        x0 = self.capex / (n * (tes2grid + pv2grid))

        lcoe = fsolve(npv, array([x0]))[0]

        return lcoe

    def capacity_factor(self, pv2grid: float, tes2grid: float):
        return (pv2grid + tes2grid) / (8.760 * self.pb_net_output)


#######################################################################################################################
#######################################################################################################################


def change_parameters(template_path: Path, file_path: Path, file_name: str, keys: list, values: list):
    """
    This function modify the desired keywords in a dck file to the desired values
    :param template_path: The full path of the template dck file to be modified.
    :param file_path: the path of where the generated dck file should be created
    :param file_name: the name of the generated dck file
    :param keys: the tagged keywords that will be changed
    :param values: the values that will replace the tagged keywords.
    :return:
    """

    # check if the template input file is a 'dck' file
    if str(template_path)[-3:] != "dck":
        raise "A 'dck' file was not given as the template argument"

    # check if the number of keys to be replaced equals to the number of given values
    if len(keys) != len(values):
        raise "The number of keys to be replaced is different from the number of given values"

    # opens the template file and copy its content to a variable 'file' and closes the template file
    with open(template_path, 'r') as template:
        file_e = template.read()

    # check for keys and replace it with the new values.
    for old, new in zip(keys, values):
        file_e = file_e.replace(old, str(new))

    file_full_path = Path(file_path, f"{file_name}.dck")
    # created the output file with the replaced values.
    with open(file_full_path, 'w') as file_out:
        file_out.write(file_e)

    return file_full_path


def run_trnsys(file_full_path: Path, trnsys_path='C:\\Trnsys17\\Exe\\TRNExe.exe'):
    """
    This function runs a TRNSYS dck file.
    :param file_full_path:
    :param trnsys_path:
    :return:
    """

    cmd = f'{trnsys_path} {file_full_path} /h'
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

########################################################################################################################
