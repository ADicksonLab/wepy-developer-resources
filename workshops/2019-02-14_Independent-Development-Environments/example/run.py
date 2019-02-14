import pickle
from copy import deepcopy

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

# wepy modules
from wepy.util.mdtraj import mdtraj_to_json_topology, json_to_mdtraj_topology
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.util.json_top import json_top_residue_df

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper, Mapper


# the runner for running dynamics and making and it's particular
# state class
from wepy.runners.openmm import OpenMMRunner, OpenMMState, OpenMMGPUWorker, UNIT_NAMES
from wepy.walker import Walker

# reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter

# configuration to help generate the reporter paths
from wepy.orchestration.configuration import Configuration

### FILL IN HERE

# Distance Metric
from wepy.resampling.distances.receptor import UnbindingDistance

# Resampler
from wepy.resampling.resamplers.wexplore import WExploreResampler

# optional: Boundary Conditions

## Parameters

# customize if desired

# BEGIN CUSTOM -------------------------------------------------------

SAVE_FIELDS = ('positions', 'box_vectors')
SPARSE_FIELDS = (('velocities', 10),)
ALL_ATOMS_SAVE_FREQ = 10
DEFAULT_N_WORKERS = 8
N_WALKERS = 4

PLATFORM = 'CPU' # Options: Reference, CPU, OpenCL, CUDA.


# resampler and distance metric parameters
MAX_N_REGIONS = (10, 10, 10, 10)
MAX_REGION_SIZES = (1, 0.5, .35, 0.25) # nanometers
PMIN = 1e-12
PMAX = 0.5

# binding site cutoff
CUTOFF = 0.8 # nm

# constants for the supporting functions
LIGAND_SEGID = 'HETA'
PROTEIN_SEGID = 'PROA'

# functions supporting the distance metric parameters at runtime
def ligand_idxs(json_topology):

    res_df = json_top_residue_df(json_topology)
    indices = res_df[res_df['segmentID'] == LIGAND_SEGID]['index'].values

    return indices

def protein_idxs(json_topology):

    res_df = json_top_residue_df(json_topology)
    indices = res_df[res_df['segmentID'] == PROTEIN_SEGID]['index'].values

    return indices

def binding_site_idxs(json_topology, coords, box_vectors, cutoff):

    # convert quantities to numbers in nanometers
    cutoff = cutoff.value_in_unit(unit.nanometer)
    coords = coords.value_in_unit(unit.nanometer)

    box_lengths, box_angles = box_vectors_to_lengths_angles(box_vectors.value_in_unit(unit.nanometer))

    # selecting ligand and protein binding site atom indices for
    # resampler and boundary conditions
    lig_idxs = ligand_idxs(json_topology)
    prot_idxs = protein_idxs(json_topology)

    # make a trajectory to compute the neighbors from
    traj = mdj.Trajectory(np.array([coords]),
                          unitcell_lengths=[box_lengths],
                          unitcell_angles=[box_angles],
                          topology=json_to_mdtraj_topology(json_topology))

    # selects protein atoms which have less than 8 A from ligand
    # atoms in the crystal structure
    neighbors_idxs = mdj.compute_neighbors(traj, cutoff, lig_idxs)

    # selects protein atoms from neighbors list
    binding_selection_idxs = np.intersect1d(neighbors_idxs, prot_idxs)

    return binding_selection_idxs

# END CUSTOM --------------------------------------------------------

# OTHERS , no need to customize
UNITS = UNIT_NAMES
INIT_WEIGHT = 1 / N_WALKERS

# MD System parameters

# method for nonbonded interactions
NONBONDED_METHOD = omma.CutoffPeriodic
# distance cutoff for non-bonded interactions
NONBONDED_CUTOFF = 1.0 * unit.nanometer

# constraints on MD calculations
MD_CONSTRAINTS = (omma.HBonds, )

RIGID_WATER = True
REMOVE_CM_MOTION = True
HYDROGEN_MASS = None

# Monte Carlo Barostat
# pressure to be maintained
PRESSURE = 1.0*unit.atmosphere
# temperature to be maintained
TEMP_UNIT = unit.kelvin
TEMPERATURE = 300.0 * TEMP_UNIT
# frequency at which volume moves are attempted
VOLUME_MOVE_FREQ = 50


# Langevin Integrator
FRICTION_COEFFICIENT = 1/unit.picosecond
# step size of time integrations
STEP_TIME = 0.002*unit.picoseconds



def run_sim(init_state_path, json_top_path, forcefield_paths,
            n_cycles, n_steps, n_workers,
            lig_ff=None,
            **kwargs):

    # add in the ligand force fields
    assert lig_ff is not None, "must give ligand forcefield"

    forcefield_paths.append(lig_ff)

    #### Wepy Orchestrator

    # load the wepy.OpenMMState
    with open(init_state_path, 'rb') as rf:
        init_state = pickle.load(rf)

    ### Apparatus

    # Runner components

    # load the JSON for the topology
    with open(json_top_path) as rf:
        json_top_str = rf.read()

    # load it with mdtraj and then convert to openmm
    mdj_top = json_to_mdtraj_topology(json_top_str)
    omm_topology = mdj_top.to_openmm()

    # we need to use the box vectors for setting the simulation up,
    # paying mind to the units
    box_vectors = init_state['box_vectors'] * init_state.box_vectors_unit

    # set the box to the last box size from equilibration
    omm_topology.setPeriodicBoxVectors(box_vectors)

    # force field parameters
    force_field = omma.ForceField(*forcefield_paths)

    # create a system using the topology method giving it a topology and
    # the method for calculation
    runner_system = force_field.createSystem(omm_topology,
                                               nonbondedMethod=NONBONDED_METHOD,
                                               nonbondedCutoff=NONBONDED_CUTOFF,
                                               constraints=MD_CONSTRAINTS,
                                               rigidWater=RIGID_WATER,
                                               removeCMMotion=REMOVE_CM_MOTION,
                                               hydrogenMass=HYDROGEN_MASS)

    # barostat to keep pressure constant
    runner_barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)
    # add it to the system
    runner_system.addForce(runner_barostat)

    # set up for a short simulation to runner and prepare
    # instantiate an integrator
    runner_integrator = omm.LangevinIntegrator(TEMPERATURE,
                                               FRICTION_COEFFICIENT,
                                               STEP_TIME)

    ## Runner
    runner = OpenMMRunner(runner_system, omm_topology, runner_integrator, platform=PLATFORM)

    ## Resampler

    # Distance Metric

    lig_idxs = ligand_idxs(json_top_str)
    prot_idxs = protein_idxs(json_top_str)
    bs_idxs = binding_site_idxs(json_top_str, init_state['positions'],
                                init_state['box_vectors'],
                                CUTOFF)


    # set distance metric
    distance_metric = UnbindingDistance(lig_idxs, bs_idxs, init_state)

    # set resampler
    resampler = WExploreResampler(distance=distance_metric,
                                   init_state=init_state,
                                   max_n_regions=MAX_N_REGIONS,
                                   max_region_sizes=MAX_REGION_SIZES,
                                   pmin=PMIN, pmax=PMAX)

    ## Boundary Conditions

    # optional: set the boundary conditions
    bc = None

    # apparatus = WepySimApparatus(runner, resampler=resampler,
    #                              boundary_conditions=bc)

    print("created apparatus")

    ## CONFIGURATION

    # the idxs of the main representation to save in the output files,
    # it is just the protein and the ligand

    # TODO optional: set the main representation atom indices
    main_rep_idxs = None


    # REPORTERS
    # list of reporter classes and partial kwargs for using in the
    # orchestrator

    hdf5_reporter_kwargs = {'main_rep_idxs' : main_rep_idxs,
                            'topology' : json_top_str,
                            'resampler' : resampler,
                            'boundary_conditions' : bc,
                            # general parameters
                            'save_fields' : SAVE_FIELDS,
                            'units' : dict(UNITS),
                            'sparse_fields' : dict(SPARSE_FIELDS),
                            'all_atoms_rep_freq' : ALL_ATOMS_SAVE_FREQ}

    # get all the reporters together. Order is important since they
    # will get paired with the kwargs
    reporter_classes = [WepyHDF5Reporter,]

    # collate the kwargs in the same order
    reporter_kwargs = [hdf5_reporter_kwargs,]

    # make the configuration with all these reporters and the default number of workers
    configuration = Configuration(n_workers=DEFAULT_N_WORKERS,
                                  reporter_classes=reporter_classes,
                                  reporter_partial_kwargs=reporter_kwargs,
                                  config_name="no-orch")

    # then instantiate them
    reporters = configuration._gen_reporters()

    print("created configuration")

    ### Initial Walkers
    init_walkers = [Walker(deepcopy(init_state), INIT_WEIGHT) for _ in range(N_WALKERS)]

    print("created init walkers")

    ### Orchestrator
    # orchestrator = Orchestrator(apparatus,
    #                             default_init_walkers=init_walkers,
    #                             default_configuration=configuration)

    ### Work Mapper
    if PLATFORM in ('OpenCL', 'CUDA'):
        # we use a mapper that uses GPUs
        work_mapper = WorkerMapper(worker_type=OpenMMGPUWorker,
                                   num_workers=n_workers)
    if PLATFORM in ('Reference', 'CPU'):
        # we just use the standard mapper
        work_mapper = Mapper

    ### Simulation Manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=bc,
                          work_mapper=work_mapper,
                          reporters=reporters)

    ### Run the simulation
    steps = [n_steps for _ in range(n_cycles)]
    sim_manager.run_simulation(n_cycles, steps)


if __name__ == "__main__":
    import sys
    import os.path as osp
    import itertools as it

    init_state = osp.realpath(sys.argv[1])
    top_json_path = osp.realpath(sys.argv[2])
    prot_ff = osp.realpath(sys.argv[4])
    solvent_ff = osp.realpath(sys.argv[5])
    n_cycles = int(sys.argv[8])
    n_steps = int(sys.argv[9])
    n_workers = int(sys.argv[10])

    # the rest of the domain specific kwargs are passed in at the end
    keys = list(it.islice(sys.argv[11:], 2))
    values = list(it.islice(sys.argv[12:], 2))
    assert len(keys) == len(values), "must be a value for every key"

    kwargs = {str(key) : str(value) for key, value in zip(keys, values)}

    run_sim(init_state, top_json_path,
            [prot_ff, solvent_ff],
            n_cycles, n_steps, n_workers,
            # domain specific kwargs
            **kwargs
    )

