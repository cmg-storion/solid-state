"""Wrappers for pymatgen functionality."""

__author__ = "Artem Dembitskiy"

# adapted from https://matgenb.materialsvirtuallab.org/2019/03/11/Interface-Reactions.html


import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.analysis.phase_diagram import (
    GrandPotentialPhaseDiagram,
    PhaseDiagram,
)
from pymatgen.analysis.interface_reactions import (
    GrandPotentialInterfacialReactivity,
    InterfacialReactivity,
)


def electrochemical_stability_window(
    entry,
    entries,
    ref_element=None,
    tol=1e-6,
    return_phase_diagram=True,
):
    """
    Calculate reduction/oxidation limits and corresponding reactions.

    Parameters
    ----------
    entry : pymatgen ComputedEntry or ComputedStructureEntry
        A material with the calculated total energy.

    entries : list of pymatgen ComputedEntry or ComputedStructureEntry
        Materials forming the chemical system of the entry material.

    ref_element : str
        Reference element (e.g. Li).

    tol : float, optional
        Numerical tolerance to find stability regions (default: 1e-6).

    return_phase_diagram : bool, optional
        Include PhaseDiagram object in output (default: True).

    Returns
    -------
    dict
        Reduction/oxidation limits and corresponding reactions.
    """

    if ref_element is None:
        raise ValueError("ref_element must be provided.")

    phase_diagram = PhaseDiagram([entry] + entries)

    ref_element_entries = [
        e for e in entries
        if e.composition.reduced_formula == ref_element
    ]
    ref_chem_pot_0 = min(
        ref_element_entries,
        key=lambda e: e.energy_per_atom,
    ).energy_per_atom

    element_profile = phase_diagram.get_element_profile(
        Element(ref_element),
        entry.composition,
    )
    reactions = pd.DataFrame(element_profile)
    reactions["voltage"] = -reactions["chempot"].values + ref_chem_pot_0

    try:
        stable_id = np.argwhere(
            abs(reactions.evolution.values) < tol
        ).ravel()[0]

        if stable_id == 0:
            window = reactions.loc[
                [stable_id, stable_id, stable_id + 1]
            ].drop(
                ["element_reference", "entries", "critical_composition"],
                axis=1,
            )
        else:
            window = reactions.loc[
                [stable_id - 1, stable_id, stable_id + 1]
            ].drop(
                ["element_reference", "entries", "critical_composition"],
                axis=1,
            )
    except Exception:
        window = None

    result = {
        "reduction_limit": (
            window.iloc[1].voltage if window is not None else None
        ),
        "oxidation_limit": (
            window.iloc[-1].voltage if window is not None else None
        ),
        "reduction_reaction": (
            window.iloc[0].reaction if window is not None else None
        ),
        "oxidation_reaction": (
            window.iloc[-1].reaction if window is not None else None
        ),
    }

    if return_phase_diagram:
        result.update(
            {
                "reactions": reactions,
                "phase_diagram": phase_diagram,
            }
        )

    return result


def chemical_mixing_energy(
    entry1,
    entry2,
    entries,
    open_element=None,
    relative_mu=None,
    norm=True,
    include_no_mixing_energy=True,
    use_hull_energy=True,
    return_phase_diagram=True,
):
    """Calculate decomposition reactions between two materials in contact."""
    comp1 = entry1.composition
    comp2 = entry2.composition
    entries = entries.copy() + [entry1, entry2]
    phase_diagram = PhaseDiagram(entries)

    if open_element:
        if relative_mu is None:
            raise ValueError(
                "relative_mu must be provided when open_element is set."
            )

        mu = phase_diagram.get_transition_chempots(
            Element(open_element)
        )[0]
        chempots = {open_element: relative_mu + mu}

        gpd = GrandPotentialPhaseDiagram(entries, chempots)

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

    col = "E$_{\textrm{rxn}}$ (eV/atom)"
    index = interface_table[col].idxmin()
    emin = interface_table[col].iloc[index]
    reaction = interface_table["Reaction"].iloc[index]

    result = {
        "e_rxn": emin,
        "reaction": reaction,
    }

    if return_phase_diagram:
        result.update(
            {
                "phase_diagram": phase_diagram,
                "profile": interface_table,
                "interface": interface,
            }
        )

    return result


def energy_above_hull(entry, entries, return_phase_diagram=True):
    """Calculate energy above the convex hull."""
    phase_diagram = PhaseDiagram([entry] + entries)

    result = {
        "e_hull": phase_diagram.get_e_above_hull(entry),
    }

    if return_phase_diagram:
        result.update({"phase_diagram": phase_diagram})

    return result
