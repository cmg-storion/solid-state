""""Pipeline for fast efficient requests from MPRester """

__author__ = 'Artem Dembitskiy'

from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry


from .data.utils import get_chemical_subsystems
from .data import DataSampler
from .thermodynamics import (
    energy_above_hull,
    electrochemical_stability_window,
    chemical_mixing_energy,
)


@dataclass
class MPWorkflow:
    api_key: str
    thermo_type: str = "GGA_GGA+U"
    cache_dir: str = "./cache"
    batch_size: int = 500


    def __post_init__(self):
        self.mpr = MPRester(self.api_key)
        self._phase_entry_cache = {}


    def query_data(self, query, exclude_elements=[]):

        """
        Wrapper for MPRester and some postfiltering

        Parameters
        ----------
        
        query: dict
            kwargs compatible with MPRester.summary.serach

        exclude_elements: set or list
            filter out entries with these elements

        Returns
        -------

        pd.DataFrame with columns ['material_id', 'chemsys', 'formula_pretty', 'energy_per_atom']
        """
        
        # collect mp_ids
        docs = self.mpr.materials.summary.search(**query)
        mp_ids = [str(d.material_id) for d in docs]

        # collect energies
        thermo_docs = self.mpr.materials.thermo.search(
            material_ids=mp_ids,
            thermo_types=[self.thermo_type],
            fields=[
                "material_id",
                "chemsys",
                "energy_per_atom",
                "formula_pretty",
            ],
        )

        rows = []
        for doc in thermo_docs:
            els = doc.chemsys.split("-")
            if any(e in exclude_elements for e in els):
                continue
            rows.append({
                "material_id": str(doc.material_id),
                "chemsys": doc.chemsys,
                "formula_pretty": doc.formula_pretty,
                "energy_per_atom": doc.energy_per_atom,
            })

        return pd.DataFrame(rows)



    def _thermo_entries_from_mpids(self, mp_ids):

        """Batched thermo data query from MP"""

        entries = []
        rows = []
        for i in tqdm(range(0, len(mp_ids), self.batch_size), desc='Getting thermo data'):
            docs = self.mpr.materials.thermo.search(
                material_ids=mp_ids[i:i+self.batch_size],
                thermo_types=[self.thermo_type],
                fields=[
                    "material_id",
                    "chemsys",
                    "formula_pretty",
                    "energy_per_atom",
                ],
            )

            
            for doc in docs:
                rows.append({
                    "material_id": str(doc.material_id),
                    "chemsys": doc.chemsys,
                    "formula_pretty": doc.formula_pretty,
                    "energy_per_atom": doc.energy_per_atom,
                })

                energy = doc.energy_per_atom * Composition(doc.formula_pretty).num_atoms
                entries.append(
                    ComputedEntry(
                        doc.formula_pretty,
                        energy,
                        entry_id=str(doc.material_id),
                        )
                    )
        return entries, pd.DataFrame(rows)
    


    def get_phase_entries(self, chemsys_list):
        
        """Get entries from list of chemical systems"""

        chemsys_key = tuple(sorted(chemsys_list))

        if chemsys_key in self._phase_entry_cache:
            return self._phase_entry_cache[chemsys_key]

        mp_ids = set()
        for i in tqdm(range(0, len(chemsys_list), self.batch_size), desc='Getting mp_ids'):
            docs = self.mpr.materials.summary.search(
                chemsys=chemsys_list[i:i+self.batch_size],
                fields=["material_id"],
            )
            mp_ids.update(str(d.material_id) for d in docs)

        entries, _ = self._thermo_entries_from_mpids(list(mp_ids))
        self._phase_entry_cache[chemsys_key] = entries
        return entries



    def electrochemical_windows(self, candidates_df, ref_element):
        """

        Calculate electrochemical stability windows for the selected candidates

        Parameters
        ----------

        candaidates_df: pd.DataFrame
            DataFrame with candidate materials.
            candidates_df must contain columns:
            ['material_id', 'chemsys', 'formula_pretty', 'energy_per_atom']

        ref_element: str
            Stability window is calculated against this element.


        Returns
        -------

        pd.DataFrame with the calculated stability regions
        """

        subsystems = set()
        for cs in candidates_df.chemsys.unique():
            subsystems.update(get_chemical_subsystems(cs))

        phase_entries = self.get_phase_entries(sorted(subsystems))
        ds = DataSampler(phase_entries)

        results = []

        for row in tqdm(
            candidates_df.itertuples(),
            total=len(candidates_df),
            desc="Calculating electrochemical stability windows",
        ):
            candidate = ComputedEntry(
                row.formula_pretty,
                row.energy_per_atom * Composition(row.formula_pretty).num_atoms,
                entry_id=row.material_id,
            )

            chemsys_entries = ds.get_entries_from_chemsys(
                candidate.composition.chemical_system
            )

            r = electrochemical_stability_window(
                candidate,
                chemsys_entries,
                ref_element=ref_element,
                return_phase_diagram=False
            )

            r["material_id"] = row.material_id

            e_hull = energy_above_hull(
                candidate,
                chemsys_entries,
                return_phase_diagram=False
            )["e_hull"]
            r["e_hull"] = e_hull

            results.append(r)

        columns = [
            "material_id",
            "reduction_limit", 
            "oxidation_limit", 
            "reduction_reaction", 
            "oxidation_reaction", 
            "e_hull",
              ]

        return pd.DataFrame(results)[columns]
        

    def interface_chemical_mixing(
        self,
        coatings_df,
        contact_df,
        open_element=None,
        relative_mu=None,
        include_no_mixing_energy=True,
    ):
        """
        Compute anode-coating interface reaction energies against solid electrolytes.

        Optimized: requests all necessary phase entries at once.

        Parameters
        ----------
        coatings_df : pd.DataFrame
            Output of screen_candidates()

        contact_df : pd.DataFrame
            Contacting material

        Returns
        -------
        pd.DataFrame
        """
        
        coating_entries = [
            ComputedEntry(
                row.formula_pretty,
                row.energy_per_atom * Composition(row.formula_pretty).num_atoms,
                entry_id=row.material_id,
            )
            for row in coatings_df.itertuples()
        ]

        contact_entries = [
            ComputedEntry(
                row.formula_pretty,
                row.energy_per_atom * Composition(row.formula_pretty).num_atoms,
                entry_id=row.material_id,
            )
            for row in contact_df.itertuples()
        ]

        all_subsystems = set()
        all_subsystems = set()
        for c in contact_entries:
            c_elements = c.composition.chemical_system.split("-")
            for coating in coating_entries:
                coating_elements = coating.composition.chemical_system.split("-")
                interface_elements = sorted(set(c_elements) | set(coating_elements))
                interface_chemsys = "-".join(interface_elements)
                all_subsystems.update(get_chemical_subsystems(interface_chemsys))

        all_phase_entries = self.get_phase_entries(sorted(all_subsystems))  
        ds = DataSampler(all_phase_entries)

        results = []
        all_subsystems = set()
        for c in contact_entries:
            c_elements = c.composition.chemical_system.split("-")
            for coating in tqdm(coating_entries, desc="Calculating chemical mixing energies"):
                coating_elements = coating.composition.chemical_system.split("-")
                interface_elements = sorted(set(c_elements) | set(coating_elements))
                interface_chemsys = "-".join(interface_elements)

                interface_entries = ds.get_entries_from_chemsys(interface_chemsys)

                r = chemical_mixing_energy(
                    coating,
                    c,
                    interface_entries,
                    norm=True,
                    include_no_mixing_energy=include_no_mixing_energy,
                    use_hull_energy=True,
                    return_phase_diagram=False,
                    open_element=open_element,
                    relative_mu=relative_mu,
                )

                r.update({
                    "material_id": coating.entry_id,
                    "reduced_formula": coating.composition.reduced_formula,
                    "contact_with": c.composition.reduced_formula,
                })

                results.append(r)       
            
        columns = [
            "material_id",
            "reduced_formula",
            "e_rxn",
            "reaction",
            "contact_with"
            ]

        return pd.DataFrame(results)[columns]
