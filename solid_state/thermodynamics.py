"""Wrappers for pymatgen's functionality"""

import numpy as np
import pandas as pd 
from pymatgen.core import Element
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.interface_reactions import InterfacialReactivity, GrandPotentialInterfacialReactivity



def electrochemical_stability_window(entry, entries, ref_element=None, tol=1e-6, return_phase_diagram=True):
    """
    Calculates reduction/oxidation limits and corresponding reactions

    Parameters
    ----------

    entry: pymatgen's ComputedEntry or ComputedStructureEntry
        A material with the calculated total energy

    ref_element: string
        Reference element, e.g. Li

    entries: list of pymatgen's ComputedEntry or ComputedStructureEntry objects
        Materials forming the chemical system of the entry material
    
    tol: float, 1e-6 by default
        numerical tolerance to find the stability regions

    return_phase_diagram: boolean, True by default
        Include pymatgen's PhaseDiagram object in the output dict

    Returns
    -------

    dictionary with reduction/oxidation limits and coresponding reactions
    """

    if ref_element is None:
        raise ValueError(f'ref_element is None')
    phase_diagram = PhaseDiagram([entry] + entries)
    ref_element_entries = [e for e in entries if e.composition.reduced_formula == ref_element]
    ref_chem_pot_0 = min(ref_element_entries, key=lambda e: e.energy_per_atom).energy_per_atom
    element_profile = phase_diagram.get_element_profile(Element(ref_element), entry.composition)
    reactions = pd.DataFrame(element_profile)
    reactions['voltage'] = -reactions['chempot'].values + ref_chem_pot_0
    try:
        stable_id = np.argwhere(abs(reactions.evolution.values) < tol).ravel()[0]
        if stable_id == 0:
            window = reactions.loc[[stable_id, stable_id, stable_id + 1]].drop(['element_reference',
                                                                                'entries',
                                                                                'critical_composition'], axis = 1)
        else:
            window = reactions.loc[[stable_id - 1, stable_id, stable_id + 1]].drop(['element_reference',
                                                                                    'entries',
                                                                                    'critical_composition'], axis = 1)
    except: 
        window = None

    result = {
            "reduction_limit": window.iloc[1].voltage if window is not None else None,
            "oxidation_limit": window.iloc[-1].voltage if window is not None else None,
            "reduction_reaction": window.iloc[0].reaction if window is not None else None,
            "oxidation_reaction": window.iloc[-1].reaction if window is not None else None,
            }
    if return_phase_diagram:
        result.update({"reactions": reactions, 'phase_diagram': phase_diagram})
    return result



def chemical_mixing_energy(entry1, entry2, entries,
                    open_element=None,
                    relative_mu=None,
                    norm=True,
                    include_no_mixing_energy=True,
                    use_hull_energy=True,
                    return_phase_diagram=True,
                    ):
        

    """
    Calculates decomposition reactions between two materials in contact

    Parameters
    ----------

    entry1: pymatgen's ComputedStructureEntry
        A material forming an interface

    entry2: pymatgen's ComputedStructureEntry
        A material forming an interface

    entries: list of pymatgen's ComputedStructureEntry objects
        Materials forming the chemical system of the interface materials

    open_element: str or None, None by default
        Open element for an open system (used for grand potential phase diagram calculations)

    relative_mu: float or None, None by default
        The chemical potential in the elemental reservoir for an open element
        Note: It must be provided if open_element is not None.
        
    norm: boolean, True by default
        Normalize to 1 the total number of atoms in the composition of reactant

    use_hull_energy: boolean, True by default
        Use the convex hull energy for a given composition for reaction energy calculation.
        if false, the energy of ground state structure will be used instead.
        note that in case when ground state can not be found for a composition,
        convex hull energy will be used associated with a warning message.

    return_phase_diagram: boolean, True by default
        Include pymatgen's PhaseDiagram, InterfacialRreactivity objects in results dict
    
    Returns
    -------

    dictionary with reduction/oxidation limits and coresponding reactions
    """

    comp1 = entry1.composition
    comp2 = entry2.composition
    entries = entries.copy() + [entry1, entry2]
    phase_diagram = PhaseDiagram(entries)

    # For an open system, include the grand potential phase diagram.
    if open_element:
        if relative_mu is None:
            raise ValueError("For open element the relative chemical" \
            "potential (relative_mu) must be provided.")
        # Get the chemical potential of the pure subtance.
        mu = phase_diagram.get_transition_chempots(Element(open_element))[0]
        # Set the chemical potential in the elemental reservoir.
        chempots = {open_element: relative_mu + mu}
        # print(chempots, mu)
        # Build the grand potential phase diagram
        gpd = GrandPotentialPhaseDiagram(entries, chempots)
        # Create InterfacialReactivity object.
        interface = GrandPotentialInterfacialReactivity(
            comp1,
            comp2,
            gpd,
            norm=norm,
            include_no_mixing_energy=include_no_mixing_energy,
            pd_non_grand=phase_diagram,
            use_hull_energy=use_hull_energy,
        )
    else:
        interface = InterfacialReactivity(
            comp1,
            comp2,
            phase_diagram,
            norm=norm,
            include_no_mixing_energy=include_no_mixing_energy,
            pd_non_grand=None,
            use_hull_energy=use_hull_energy,
        )
    interface_table = interface.get_dataframe()

    index = interface_table['E$_{\textrm{rxn}}$ (eV/atom)'].idxmin()
    emin = interface_table['E$_{\textrm{rxn}}$ (eV/atom)'].values[index]
    reaction = (interface_table['Reaction'].iloc[index])

    result = {'e_rxn': emin, 'reaction': reaction}
    if return_phase_diagram:
        result.update({'phase_diagram': phase_diagram})
        result.update({'profile': interface_table})
        result.update({'interface': interface})
    return result



def energy_above_hull(entry, entries, return_phase_diagram=True):
    """
    Calculates energy above the convex hull

    Parameters
    ----------

    entry: pymatgen's ComputedEntry or ComputedStructureEntry
        A material with the calculated total energy

    entries: list of pymatgen's ComputedEntry or ComputedStructureEntry objects
        Materials forming the chemical system of the entry material

    return_phase_digram: boolean, True by default
        Include pymatgen's PhaseDiagram in the output dict
    

    Returns
    -------

    dictionary with e_hull value
    """

    phase_diagram = PhaseDiagram([entry] + entries)
    result = {
            "e_hull": phase_diagram.get_e_above_hull(entry)
            }
    if return_phase_diagram:
        result.update({"phase_diagram": phase_diagram})
    return result


