"""Wrappers for the ions and ASE libraries"""

import os
import numpy as np
import pandas as pd
from ase.io import write
from ase.mep import NEB
from ase.constraints import FixAtoms
from ase.build.tools import sort
from ase.optimize import FIRE
from ions import Edge, Percolator



def find_unique_edges(atoms,
                      mobile_specie,
                      upper_bound=8.0,
                      bottleneck_radius=0.5,
                      method='naive',
                      ):
    """Find unique edges forming percolating network of a mobile specie
    
    Parameters
    ----------

    atoms: ase.Atoms
        Atomic structure

    mobile_specie: str
        Chemical element

    upper_bound: float, 8.0 by default
        Upper bound for searching nearest neighbors in mobile sublattice,
        i.e. maximum edge length

    bottleneck_radius: float, 0.5 by default
        Minimum allowed distance from the edge to the framework atoms.
        Edges with the distance < bottleneck_radius are rejected

    method: str, "naive" by default
        Method for finding symmetrically equivalent edges.
        Can be "naive" or "symop"

    Returns
    -------

    edges: list of edges
        Each edge is represented by 5 x 1 array with:
        source, target, offset_x, offset_y, offset_z 

    mincut: float
        Minimum edge length required to form the percolaiton network

    maxdim: int
        Maximum percolation dimensionality formed by the edges

    """

    pl = Percolator(atoms,
                    mobile_specie=mobile_specie,
                    upper_bound=upper_bound
                    )
    mincut, maxdim = pl.mincut_maxdim(bottleneck_radius)
    edges = pl.unique_edges(mincut, bottleneck_radius, method=method)
    return edges, mincut, maxdim



def prepare_linear_trajectories(atoms,
                                edges,
                                supercell_size=8.0,
                                n_images=7,
                                center=True
                                ):
    
    """Prepare linear guess trajectories vacancy migration for the provided edges
    
    Parameters
    ----------

    atoms: ase.Atoms
        Atomic structure

    edges: np.array or list of edges
        List of edges

    supercell_size: float, 8.0 by default
        Minimum height of the supercell

    n_images: int, 7 by default
        Number of images in the trajectory including end points

    center: boolean, True by default
        Shift trajectory to the center of the supercell

    Returns
    -------
    trajectories: list of list of ase.Atoms
        A list containing multiple ion migration trajectories.
        
        Each trajectory is a list of ase.Atoms objects representing
        intermediate configurations along the migration path.
        
        Each Atoms object has a boolean array attribute named "moving"
        where:
        - Exactly one element is True (identifying the mobile ion)
        - All other elements are False (static ions)
    """


    trajectories = []
    for edge in edges:
        source, target, offset = edge[0], edge[1], edge[2:]
        edge = Edge(atoms, source, target, offset)
        superedge = edge.superedge(r_cut=supercell_size)
        images = superedge.interpolate(n_images=n_images, center=center)
        trajectories.append(images)
    return trajectories



def idpp_preconditioning(trajectories, constrain_atoms=True):

    """
    Precondintion trajectory using IDPP method

    Parameters
    ----------

    trajectories: list of trajectories
        List of trajectories

    constrain_atoms: boolean, True by default
        Fix non-migrating ions

    Returns
    -------

    preconditioned trajectories

    """

    for trajectory in trajectories:
        if constrain_atoms:
            fixed_ions = np.where(trajectory[0].arrays['moving']==False)[0].ravel()
            c = FixAtoms(indices=fixed_ions)
            for image in trajectory:
                image.set_constraint(c)
            neb = NEB(trajectory)
            neb.interpolate('idpp')
            for image in trajectory:
                del image.constraints
        else:
            neb = NEB(trajectory)
            neb.interpolate('idpp')

    return trajectories



def neb_optimize(trajectory,
                 calculators,
                 optimizer=None,
                 submitdir=None,
                 n_steps=200,
                 fmax=0.1,
                 relax_endpoints=True,
                 verbose=True,
                 ):
    

    """
    Optimize trajectory using climbing NEB method with improved tangent estimate

    Parameters
    ----------

    trajectory: list of ase.Atoms
        Trajectory

    calculators: list of ase compatible calculators
        List of solver, len(calculators) must be >= len(trajectory)

    optimizer: ase.optimize optimizer or None
        Optimizer, will use FIRE if None

    submitdir: str
        Path to save calculations
    
    n_steps: int, 200 by default
        Maximum number of optimization steps

    f_max: float, 0.1 by default
        Force convergence criteria
    
    relax_endpoints: boolean, True by default
        Optimize end points of the trajectory

    verbose: boolean, True by default
        verbosity

    Returns
    -------

    ase's NEB optimizable and maximum NEB-force

    """

    assert len(trajectory) <= len(calculators)

    
    optimizer = FIRE if optimizer is None else optimizer


    for image, calc in zip(trajectory, calculators):
        image.calc = calc

    if submitdir is not None:
        os.makedirs(submitdir, exist_ok=True)
        write(f'{submitdir}/trajectory_init.xyz', trajectory)

    if relax_endpoints:
        if verbose:
            print(' Optimizing source')
        qn_source = optimizer(trajectory[0],
                         logfile=None if submitdir is None else f'{submitdir}/qn_source.log' ,
                         trajectory=None if submitdir is None else f'{submitdir}/qn_source.xyz',
                         )
        qn_source.run(fmax=fmax, steps=n_steps)

        if verbose:
            print(' Optimizing target')
        qn_target = optimizer(trajectory[-1],
                         logfile=None if submitdir is None else f'{submitdir}/qn_target.log' ,
                         trajectory=None if submitdir is None else f'{submitdir}/qn_target.xyz',
                         )
        qn_target.run(fmax=fmax, steps=n_steps)
    
    neb = NEB(trajectory,
              parallel=False,
              k=5.0, 
              climb=True,
              method='improvedtangent'
              )
    if verbose:
        print(' Optimizing string\n')
    qn = optimizer(neb,
                   logfile=None if submitdir is None else f'{submitdir}/qn_neb.log' ,
                   trajectory=None if submitdir is None else f'{submitdir}/neb.xyz',
                   )

    max_force_list = []
    for i, _ in enumerate(qn.irun(fmax=fmax, steps=n_steps)):
        forces = [abs(im.get_forces()).max() for im in qn.optimizable.neb.iterimages()]
        step = '{:05}'.format(i)
        max_force_list.append(max(forces))
        if submitdir is not None:
            write(f'{submitdir}/band_optim_step_{step}.traj', qn.optimizable.neb.images)

    forces = neb.get_forces()
    max_force = np.sqrt((forces ** 2).sum(axis=1).max())

    max_neb_force = np.sqrt((neb.get_forces()**2).sum(axis=1).max())
    max_source_force = np.sqrt((neb.images[0].get_forces()**2).sum(axis=1).max())
    max_target_force = np.sqrt((neb.images[-1].get_forces()**2).sum(axis=1).max())
    max_force = max([max_neb_force, max_source_force, max_target_force])
    _ = [image.info.update({'max_neb_force': max_force}) for image in qn.optimizable.neb.images]
    if submitdir is not None:
        write(f'{submitdir}/trajectory_relaxed.xyz', qn.optimizable.neb.images)

    return neb, max_force



def neb_percolation_barriers(atoms,
                              mobile_specie=None,
                              upper_bound=8.0,
                              bottleneck_radius=0.5,
                              method='naive',
                              use_idpp=True,
                              constrain_atoms=True,
                              calculators=None,
                              optimizer=None,
                              submitdir=None,
                              n_steps=500,
                              fmax=0.01,
                              n_images=7,
                              center=True,
                              supercell_size=8.0,
                              relax_endpoints=True,
                              min_max=True,
                              verbose=True,
                              ):
    
    """
    Find percolation barriers of mobile species in a given ase.Atoms object

    Parameters
    ----------

    atoms: ase.Atoms
        Atomic structure

    mobile_specie: str
        Chemical element

    upper_bound: float, 8.0 by default
        Upper bound for searching nearest neighbors in mobile sublattice,
        i.e. maximum edge length

    bottleneck_radius: float, 0.5 by default
        Minimum allowed distance from the edge to the framework atoms.
        Edges with the distance < bottleneck_radius are rejected

    method: str, "naive" by default
        Method for finding symmetrically equivalent edges.
        Can be "naive" or "symop"

    submitdir: str
        Path to save calculations

    calculators: ase compatible calculators
        List of solvers

    optimizer: ase.optimize optimizer or None
        Optimizer, will use FIRE if None

    use_idpp: boolean, True by default
        Use IDPP method for preconditioning

    constrain_atoms: boolean, True by default
        fix non-migrating ions
        Note: used only when use_idpp=True

    n_steps: int, 500 by default
        Maximum number of optimization steps

    f_max: float, 0.01 by default
        Force convergence criteria
    
    relax_endpoints: boolean, True by default
        Optimize end points of the trajectory

    supercell_size: float, 8.0 by default
        Minimum height of the supercell

    n_images: int, 7 by default
        Number of images in the trajectory including end points

    center: boolean, True by default
        Shift trajectory to the center of the supercell

    min_max: bool, True by default

        If True, use the minimum and maximum energy states along each migration 
        profile to calculate percolation barriers.
        
        If False, calculate migration barriers as (maximum - minimum) for each 
        edge individually, assuming all profiles share the same baseline energy 
        (i.e., the same minimum energy state).



    verbose: boolean, True by default
        Print out info on intermediate steps

    Returns
    -------
    dict with e1d, e2d, e3d percolation barriers
    and max_force for the obtained percolation network
    """
    
    
    pl = Percolator(atoms, mobile_specie=mobile_specie, upper_bound=upper_bound)
    mincut, maxdim = pl.mincut_maxdim(bottleneck_radius)
    edges = pl.unique_edges(mincut, bottleneck_radius, method=method)

    if verbose:
        print('Unique edges:')
        print(edges, '\n')
    
    trajectories = prepare_linear_trajectories(atoms,
                                               edges,
                                               supercell_size=supercell_size,
                                               n_images=n_images,
                                               center=center)
    
    if use_idpp:
        print('Using IDPP preconditioning\n')
        trajectories = idpp_preconditioning(trajectories, constrain_atoms=constrain_atoms)

    emins, emaxs = [], []
    max_forces = []
    for idx, (edge, trajectory) in enumerate(zip(edges, trajectories)):    
        edge_name = f"edge_id_{idx}_notation_" + "_".join(map(str, edge))
        edge_submitdir = f'{submitdir}/{edge_name}.neb' if submitdir is not None  else None
        
        print(f'({idx}/{len(edges)}) NEB optimization for {edge_name}:')
        neb, max_force = neb_optimize(trajectory,
                              calculators,
                              optimizer,
                              submitdir=edge_submitdir,
                              relax_endpoints=relax_endpoints,
                              fmax=fmax,
                              n_steps=n_steps,
                              verbose=verbose
                              )
        profile = np.array([im.get_potential_energy() for im in neb.images])

        emins.append(min(profile))
        emaxs.append(max(profile))
        max_forces.append(max_force)
    
    if min_max:
        barriers = pl.find_percolation_barriers(mincut, bottleneck_radius, emins, emaxs, method='naive')
    else:
        migration_barriers = np.array(emaxs) - np.array(emins)
        baseline_emins = np.zeros(len(migration_barriers))
        barriers = pl.find_percolation_barriers(mincut, bottleneck_radius, baseline_emins, migration_barriers, method='naive')
    barriers.update({'max_force': max(max_forces)})
    
    if submitdir is not None:
        pd.DataFrame(barriers, index =[0]).to_csv(f'{submitdir}/barriers.csv', index=False)
    
    return barriers 



def prepare_vasp_neb_folders(trajectory, path):

    """
    Prepare folders with POSCAR files for NEB calculations in VASP.
    
    Parameters
    ----------
    trajectory: list of  ase.Atoms
        Trajectory

    path: str
        path to save files

    Returns
    -------
    nothing

    """

    os.makedirs(f'{path}', exist_ok=True)
    trajectory_new = []
    sort_ids = None
    for i, image in enumerate(trajectory):
        
        # sort first image and use sort_ids
        if sort_ids is None:

            image.set_array('sort_index', np.arange(len(image)))
            image = sort(image)
            sort_ids = image.copy().arrays['sort_index']
            df = pd.DataFrame()
            df['index'] = np.arange(len(sort_ids))
            df['sort_index'] = sort_ids
            df.to_csv(f'{path}/sort_index.csv', index = False)
        else: 
            image = image[[i for i in sort_ids]]
            image.set_array('sort_index', sort_ids)

        trajectory_new.append(image)
        image_folder = str(i).zfill(2)
        os.makedirs(f'{path}/{image_folder}', exist_ok=True)
        write(f'{path}/{image_folder}/POSCAR', image, format = 'vasp')
    write(f'{path}/trajectory_init.xyz', trajectory_new, format = 'extxyz')