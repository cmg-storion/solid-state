"""Wrappers for the ions and ASE libraries"""

__author__ = 'Artem Dembitskiy'


import os
import numpy as np
import pandas as pd

from ase.io import write
from ase.mep import NEB
from ase.constraints import FixAtoms
from ase.build.tools import sort
from ase.optimize import FIRE

from ions import Edge, Percolator


def find_unique_edges(
    atoms,
    mobile_specie,
    upper_bound=8.0,
    bottleneck_radius=0.5,
    method='naive',
):
    """
    Find unique edges forming percolating network of a mobile specie.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    
    mobile_specie : str
        Chemical element.
    
    upper_bound : float, optional
        Upper bound for searching nearest neighbors in mobile sublattice,
        i.e., maximum edge length. Default is 8.0.
    
    bottleneck_radius : float, optional
        Minimum allowed distance from the edge to the framework atoms.
        Edges with distance < bottleneck_radius are rejected. Default is 0.5.
    
    method : str, optional
        Method for finding symmetrically equivalent edges.
        Can be "naive" or "symop". Default is "naive".
    
    Returns
    -------
    edges : list of edges
        Each edge is represented by 5 x 1 array with:
        source, target, offset_x, offset_y, offset_z.
    
    mincut : float
        Minimum edge length required to form the percolation network.
    
    maxdim : int
        Maximum percolation dimensionality formed by the edges.
    """
    
    pl = Percolator(atoms, mobile_specie=mobile_specie, upper_bound=upper_bound)
    mincut, maxdim = pl.mincut_maxdim(bottleneck_radius)
    edges = pl.unique_edges(mincut, bottleneck_radius, method=method)
    
    return edges, mincut, maxdim


def prepare_linear_trajectories(
    atoms,
    edges,
    supercell_size=8.0,
    n_images=7,
    center=True,
):
    """
    Prepare linear guess trajectories for vacancy migration for provided edges.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    
    edges : np.array or list of edges
        List of edges.
    
    supercell_size : float, optional
        Minimum height of the supercell. Default is 8.0.
    
    n_images : int, optional
        Number of images in the trajectory including end points. Default is 7.
    
    center : bool, optional
        Shift trajectory to the center of the supercell. Default is True.
    
    Returns
    -------
    trajectories : list of list of ase.Atoms
        A list containing multiple ion migration trajectories.
        
        Each trajectory is a list of ase.Atoms objects representing
        intermediate configurations along the migration path.
        
        Each Atoms object has a boolean array attribute named "moving" where:
        - Exactly one element is True (identifying the mobile ion)
        - All other elements are False (static ions)
    """
    
    trajectories = []
    
    for edge in edges:
        source, target, offset = edge[0], edge[1], edge[2:]
        edge_obj = Edge(atoms, source, target, offset)
        superedge = edge_obj.superedge(r_cut=supercell_size)
        images = superedge.interpolate(n_images=n_images, center=center)
        trajectories.append(images)
    
    return trajectories


def idpp_preconditioning(trajectories, constrain_atoms=True):
    """
    Precondition trajectory using IDPP method.
    
    Parameters
    ----------
    trajectories : list of trajectories
        List of trajectories.
    
    constrain_atoms : bool, optional
        Fix non-migrating ions. Default is True.
    
    Returns
    -------
    list
        Preconditioned trajectories.
    """
    
    for trajectory in trajectories:
        if constrain_atoms:
            fixed_ions = np.where(trajectory[0].arrays['moving'] == False)[0].ravel()
            constraint = FixAtoms(indices=fixed_ions)
            
            for image in trajectory:
                image.set_constraint(constraint)
            
            neb = NEB(trajectory)
            neb.interpolate('idpp')
            
            for image in trajectory:
                del image.constraints
        else:
            neb = NEB(trajectory)
            neb.interpolate('idpp')
    
    return trajectories


def neb_optimize(
    trajectory,
    calculators,
    optimizer=None,
    submitdir=None,
    n_steps=200,
    fmax=0.1,
    relax_endpoints=True,
    verbose=True,
):
    """
    Optimize trajectory using climbing NEB method with improved tangent estimate.
    
    Parameters
    ----------
    trajectory : list of ase.Atoms
        Trajectory.
    
    calculators : list
        List of ASE-compatible calculators. len(calculators) must be >= len(trajectory).
    
    optimizer : ase.optimize optimizer, optional
        Optimizer. Uses FIRE if None.
    
    submitdir : str, optional
        Path to save calculations.
    
    n_steps : int, optional
        Maximum number of optimization steps. Default is 200.
    
    fmax : float, optional
        Force convergence criteria. Default is 0.1.
    
    relax_endpoints : bool, optional
        Optimize end points of the trajectory. Default is True.
    
    verbose : bool, optional
        Verbosity. Default is True.
    
    Returns
    -------
    neb : ase.NEB
        NEB optimizable object.
    
    max_force : float
        Maximum NEB-force.
    """
    
    assert len(trajectory) <= len(calculators), \
        "Number of calculators must be >= number of images in trajectory"
    
    optimizer = FIRE if optimizer is None else optimizer
    
    for image, calc in zip(trajectory, calculators):
        image.calc = calc
    
    if submitdir is not None:
        os.makedirs(submitdir, exist_ok=True)
        write(f'{submitdir}/trajectory_init.xyz', trajectory)
    
    if relax_endpoints:
        if verbose:
            print(' Optimizing source')
        
        qn_source = optimizer(
            trajectory[0],
            logfile=None if submitdir is None else f'{submitdir}/qn_source.log',
            trajectory=None if submitdir is None else f'{submitdir}/qn_source.xyz',
        )
        qn_source.run(fmax=fmax, steps=n_steps)
        
        if verbose:
            print(' Optimizing target')
        
        qn_target = optimizer(
            trajectory[-1],
            logfile=None if submitdir is None else f'{submitdir}/qn_target.log',
            trajectory=None if submitdir is None else f'{submitdir}/qn_target.xyz',
        )
        qn_target.run(fmax=fmax, steps=n_steps)
    
    neb = NEB(
        trajectory,
        parallel=False,
        k=5.0,
        climb=True,
        method='improvedtangent',
    )
    
    if verbose:
        print(' Optimizing string\n')
    
    qn = optimizer(
        neb,
        logfile=None if submitdir is None else f'{submitdir}/qn_neb.log',
        trajectory=None if submitdir is None else f'{submitdir}/neb.xyz',
    )
    
    max_force_list = []
    
    for i, _ in enumerate(qn.irun(fmax=fmax, steps=n_steps)):
        forces = [abs(image.get_forces()).max() 
                 for image in qn.optimizable.neb.iterimages()]
        step = f'{i:05d}'
        max_force_list.append(max(forces))
        
        if submitdir is not None:
            write(f'{submitdir}/band_optim_step_{step}.traj', 
                  qn.optimizable.neb.images)
    
    max_neb_force = np.sqrt((neb.get_forces() ** 2).sum(axis=1).max())
    max_source_force = np.sqrt((neb.images[0].get_forces() ** 2).sum(axis=1).max())
    max_target_force = np.sqrt((neb.images[-1].get_forces() ** 2).sum(axis=1).max())
    
    max_force = max([max_neb_force, max_source_force, max_target_force])
    
    for image in qn.optimizable.neb.images:
        image.info.update({'max_neb_force': max_force})
    
    if submitdir is not None:
        write(f'{submitdir}/trajectory_relaxed.xyz', qn.optimizable.neb.images)
    
    return neb, max_force


def neb_percolation_barriers(
    atoms,
    mobile_specie=None,
    upper_bound=8.0,
    bottleneck_radius=0.5,
    method='naive',
    use_idpp=True,
    constrain_atoms=True,
    calculators=None,
    optimizer=None,
    submitdir=None,
    n_steps=200,
    fmax=0.1,
    n_images=7,
    center=True,
    supercell_size=8.0,
    relax_endpoints=True,
    min_max=True,
    verbose=True,
):
    """
    Find percolation barriers of mobile species in a given ase.Atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    
    mobile_specie : str
        Chemical element.
    
    upper_bound : float, optional
        Upper bound for searching nearest neighbors in mobile sublattice.
        Default is 8.0.
    
    bottleneck_radius : float, optional
        Minimum allowed distance from edge to framework atoms.
        Default is 0.5.
    
    method : str, optional
        Method for finding symmetrically equivalent edges.
        Default is "naive".
    
    use_idpp : bool, optional
        Use IDPP method for preconditioning. Default is True.
    
    constrain_atoms : bool, optional
        Fix non-migrating ions (used only when use_idpp=True). Default is True.
    
    calculators : list, optional
        ASE-compatible calculators.
    
    optimizer : ase.optimize optimizer, optional
        Optimizer. Uses FIRE if None.
    
    submitdir : str, optional
        Path to save calculations.
    
    n_steps : int, optional
        Maximum number of optimization steps. Default is 200.
    
    fmax : float, optional
        Force convergence criteria. Default is 0.1.
    
    n_images : int, optional
        Number of images in trajectory including end points. Default is 7.
    
    center : bool, optional
        Shift trajectory to center of supercell. Default is True.
    
    supercell_size : float, optional
        Minimum height of supercell. Default is 8.0.
    
    relax_endpoints : bool, optional
        Optimize end points of trajectory. Default is True.
    
    min_max : bool, optional
        If True, use min and max energy states along each migration profile
        to calculate percolation barriers.
        
        If False, calculate migration barriers as (maximum - minimum) for each
        edge individually, assuming all profiles share same baseline energy.
        Default is True.
    
    verbose : bool, optional
        Print info on intermediate steps. Default is True.
    
    Returns
    -------
    dict
        Dictionary with e1d, e2d, e3d percolation barriers and max_force
        for the obtained percolation network.
    """
    
    pl = Percolator(atoms, mobile_specie=mobile_specie, upper_bound=upper_bound)
    mincut, maxdim = pl.mincut_maxdim(bottleneck_radius)
    edges = pl.unique_edges(mincut, bottleneck_radius, method=method)
    
    if verbose:
        print('Unique edges:')
        print(edges, '\n')
    
    trajectories = prepare_linear_trajectories(
        atoms,
        edges,
        supercell_size=supercell_size,
        n_images=n_images,
        center=center,
    )
    
    if use_idpp:
        print('Using IDPP preconditioning\n')
        trajectories = idpp_preconditioning(trajectories, constrain_atoms=constrain_atoms)
    
    emins, emaxs = [], []
    max_forces = []
    
    for idx, (edge, trajectory) in enumerate(zip(edges, trajectories)):
        edge_name = f"edge_id_{idx}_notation_" + "_".join(map(str, edge))
        edge_submitdir = f'{submitdir}/{edge_name}.neb' if submitdir is not None else None
        
        if verbose:
            print(f'({idx + 1}/{len(edges)}) NEB optimization for {edge_name}:')
        
        neb, max_force = neb_optimize(
            trajectory,
            calculators,
            optimizer,
            submitdir=edge_submitdir,
            relax_endpoints=relax_endpoints,
            fmax=fmax,
            n_steps=n_steps,
            verbose=verbose,
        )
        
        profile = np.array([image.get_potential_energy() for image in neb.images])
        
        emins.append(min(profile))
        emaxs.append(max(profile))
        max_forces.append(max_force)
    
    if min_max:
        barriers = pl.find_percolation_barriers(
            mincut, bottleneck_radius, emins, emaxs, method='naive'
        )
    else:
        migration_barriers = np.array(emaxs) - np.array(emins)
        baseline_emins = np.zeros(len(migration_barriers))
        barriers = pl.find_percolation_barriers(
            mincut, bottleneck_radius, baseline_emins, migration_barriers, method='naive'
        )
    
    barriers.update({'max_force': max(max_forces)})
    
    if submitdir is not None:
        pd.DataFrame(barriers, index=[0]).to_csv(f'{submitdir}/barriers.csv', index=False)
    
    return barriers


def prepare_vasp_neb_folders(trajectory, path):
    """
    Prepare folders with POSCAR files for NEB calculations in VASP.
    
    Parameters
    ----------
    trajectory : list of ase.Atoms
        Trajectory.
    
    path : str
        Path to save files.
    
    Returns
    -------
    None
    """
    
    os.makedirs(f'{path}', exist_ok=True)
    trajectory_new = []
    sort_ids = None
    
    for i, image in enumerate(trajectory):
        
        # Sort first image and use sort_ids for consistency
        if sort_ids is None:
            image.set_array('sort_index', np.arange(len(image)))
            image = sort(image)
            sort_ids = image.copy().arrays['sort_index']
            
            df = pd.DataFrame()
            df['index'] = np.arange(len(sort_ids))
            df['sort_index'] = sort_ids
            df.to_csv(f'{path}/sort_index.csv', index=False)
        else:
            image = image[[i for i in sort_ids]]
            image.set_array('sort_index', sort_ids)
        
        trajectory_new.append(image)
        image_folder = str(i).zfill(2)
        os.makedirs(f'{path}/{image_folder}', exist_ok=True)
        write(f'{path}/{image_folder}/POSCAR', image, format='vasp')
    
    write(f'{path}/trajectory_init.xyz', trajectory_new, format='extxyz')