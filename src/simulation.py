"""
Wraps an external simulator (Morpheus, xTB, etc.) as a unified API,
with proper progress logging and unit conversion.
"""

# Force xTB parameter path so Morpheus doesn't default to /root
import os

os.environ["XTBPATH"] = os.environ.get("XTBPATH", "/usr/local/share/xtb")

import logging

# Optional dependency - graceful handling
try:
    from morpheus.molecule import Smiles
    from morpheus.reaction import ReactionTemplate, Reaction
    from morpheus.simulation import Simulation, SimulationOptions
    from morpheus.simulation.options import (
        ConformerSearchMethod,
        ConformerSearchOptions,
        GFNLevel,
    )
    from morpheus.utils.units import EnergyUnit, convert

    HAS_MORPHEUS = True
except ImportError:
    HAS_MORPHEUS = False


class EnergySimulator:
    def __init__(self, options=None):
        self.options = options
        self.logger = logging.getLogger(__name__)
        # log XTBPATH for confirmation
        xtbpath = os.environ.get("XTBPATH")
        self.logger.info(f"Set XTBPATH to '{xtbpath}'")

        if not HAS_MORPHEUS:
            self.logger.warning(
                "Morpheus not available. Simulator will only work with pre-cached energy values. "
                "Ensure all SMILES are present in the energy cache file."
            )

    def compute(self, smiles: str, index: int = None, total: int = None) -> float:
        if not HAS_MORPHEUS:
            # Fallback mode - only works with cached values
            if index is not None and total is not None:
                self.logger.error(
                    f"Cannot simulate {index}/{total}: {smiles} - morpheus not available"
                )
            else:
                self.logger.error(f"Cannot simulate: {smiles} - morpheus not available")
            raise RuntimeError(
                "Morpheus not available. This simulator can only be used with pre-computed cache values. "
                "All SMILES must be present in the energy cache file."
            )

        if index is not None and total is not None:
            self.logger.info(f"Simulating {index}/{total}: {smiles}")
        else:
            self.logger.info(f"Simulating: {smiles}")

        template = ReactionTemplate(
            r"[#6-;v3;D2:1]~1~[#7+;v4:2]~[$([#6;v4]),$([#7;v3]):3]"
            r"~[$([#6;v4;X3]),$([#7;v3;X2]):4]~[$([#7;v3]),$([#16;v2]):5]1>>"
            r"[#6+0;v4;X3:1](C(=O)[O-])-1=[#7+;v4;X3:2]"
            r"-[$([#6;v4;X3]),$([#7;v3;X2]):3]=[$([#6;v4;X3]),$([#7;v3;X2]):4]"
            r"-[$([#7;v3]),$([#16;v2]):5]1"
        )
        sim_opts = SimulationOptions(
            gfn_level=GFNLevel.GFN2,
            xtb_cores=12,
            conformer_search_options=ConformerSearchOptions(
                method=ConformerSearchMethod.RDKIT, rdkit_level=200
            ),
        )
        sim = Simulation(sim_opts)
        substrate = Smiles(smiles)
        co2_ref = -10.317253938960
        reaction = Reaction(template)
        reaction.add_reactants([substrate])
        reaction.run_reaction()
        dg = sim.calculate_delta_g(reaction.products[0])
        dg -= co2_ref
        kj = float(convert(dg, EnergyUnit.Eh, EnergyUnit.kJMol))
        self.logger.info(f"Computed energy for {smiles}: {kj:.2f} kJ/mol")
        return kj
