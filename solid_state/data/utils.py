import itertools
from pymatgen.entries.computed_entries import ComputedEntry



U_values = {
'Co': 3.32,
'Cr': 3.7,
'Fe': 5.3,
'Mn': 3.9,
'Mo': 4.38,
'Ni': 6.2,
'V': 3.25,
'W': 6.2
}



def get_chemical_subsystems(chemsys):
    """
    Get list of all possible chemical subsystems of the provided chemical system

    Parameters
    ----------

    chemsys: str
        Chemical system in a format of elements separated by - symbol, e.g. "Li-P-O"

    Returns
    -------
    list of chemical systems

    """
    elements = chemsys.split("-")
    chemsys_list = []
    for i in range(1, len(elements)+1):
        elements_subset = [list(x) for x in itertools.combinations(elements, i)]
        for subset in elements_subset:
            subset.sort(key=lambda x: x.lower())
        chemsys_list.extend(elements_subset)
    return ['-'.join(chemsys) for chemsys in chemsys_list]


def set_MP2020_settings(entry, run_type=None):
    """
    Get dicitonary with a default MP2020 computational settings

    Parameters
    ----------

    entry: ComputedEntry or ComputedStructureEntry
        Entry to set

    run_type: str or None
        If None will use GGA or GGA+U depending on the chemical composition
    """

    # from pymatgen.entries.compatibility.MaterialsProject2020Compatibility.u_settings
    U_values = {
    'Co': 3.32,
    'Cr': 3.7,
    'Fe': 5.3,
    'Mn': 3.9,
    'Mo': 4.38,
    'Ni': 6.2,
    'V': 3.25,
    'W': 6.2
    }
    composition = entry.composition
    hubbards = {}
    elements = [str(element) for element in entry.composition.elements]
    
    if len(set(['O', 'F']) - set(elements)) < 2:
        for element in composition.elements:
            if str(element) in U_values.keys():
                hubbards.update({str(element): U_values[str(element)]})
    if len(hubbards) > 0:
        run_type = "GGA+U" if run_type is None else run_type
    else:
        run_type = "GGA" if run_type is None else run_type
    
    if len(set(['O', 'H']) - set(elements)) == 0:
        ox_type = 'hydroxide'

    # not used for now
    common_peroxides = "Li2O2 Na2O2 K2O2 Cs2O2 Rb2O2 BeO2 MgO2 CaO2 SrO2 BaO2".split()
    common_superoxides = "LiO2 NaO2 KO2 RbO2 CsO2".split()
    ozonides = "LiO3 NaO3 KO3 NaO5".split()


    parameters = {"hubbards": hubbards, "run_type": run_type}
    entry.parameters = parameters



def prepare_computed_entries(compositions, energies, material_ids=None):
    
    """
    Prepare CoputedEntry objects from a list of compositions and energies

    Parameters
    ----------

    compositions: pymatgen's Composition
        Chemical composition of the structure

    energy: float
        Computed energy per provided chemical composition

    material_ids: list of str or None, None by default
        Identifiers of the materials. If None will assign IDs from 0 to len(energies)

    Returns
    -------
    
    """

    assert len(compositions) == len(energies)

    if material_ids is None:
        material_ids = [f'ID-{i}' for i in range(len(compositions))]

    entries = [ComputedEntry(comp, energy = e, entry_id = mp_id)\
                     for comp, e, mp_id in zip(compositions, energies, material_ids)]
    
    return entries
