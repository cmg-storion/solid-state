import pandas as pd
from .utils import get_chemical_subsystems

class DataSampler:
    """
    Data sampler used for thermodynamic calculations for large datasets

    """
    def __init__(self, entries): 

        """
        Initialization

        Parameters
        ----------

        entries: list of pymatgen's ComputedEntry of ComputedStructureEntry
            entries collected for the thermodynamic calculations

        """
        self.entries = entries
        self.n_entries = len(entries)
        self._process()

    def _process(self):
        chemsys_list = []
        for entry in self.entries:
            chemsys_list.append(entry.composition.chemical_system)
        self._table = pd.DataFrame({'entry': self.entries, 'chemsys': chemsys_list})

    def get_entries_from_chemsys(self, chemsys):
        """
        Get subset of entries from the chemical system of interest

        Parameters
        ----------
        chemsys: str
            chemical system given as elements separated by "-" sign, e.g. "Li-Co-O"

        Returns
        -------
        entries: list of pymatgen's ComputedEntry or ComputedStructureEntry
            entries forming provided chemical system sampled from the initial pull of entries 
        """
        subsystems = get_chemical_subsystems(chemsys)
        return self._table[self._table.chemsys.isin(subsystems)].entry.values.tolist()

    def __repr__(self):
        return f"DataSampler(n_entries = {self.n_entries})"