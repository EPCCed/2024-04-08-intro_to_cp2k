#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from aml.io import cp2k
from aml.structures import Structures
import aml


# Set random number seed for reproducability
np.random.seed(0)

def get_cell(folder, project = "sys"):
    cell_df = pd.read_csv(f"{folder}/{project}-1.cell", sep = "[ ]{2,}", engine='python')
    names = [name for name in list(cell_df) if name.endswith("[Angstrom]")]
    cell_df = cell_df[names]
    cells = cell_df.values.reshape((len(cell_df), 3, 3))
    return cells

def read_structures(folder = "../ref_data/nnp_ref", project = "sys"):

    # location of data
    fn_positions =  f'{folder}/{project}-pos-1.xyz'
    fn_forces =  f'{folder}/{project}-frc-1.xyz'
    cells = get_cell(folder, project = project)

    # stride through trajectory
    stride_trj = 1

    print('Reading structures')
    print('------------------')
    print()
    print(f'Directory: {folder}')
    print(f'Stride: {stride_trj}')
    print()

    frames = cp2k.read_frames_cp2k(
        fn_positions = fn_positions,
        fn_forces = fn_forces,
        cells = cells
    )
    structures = Structures.from_frames(frames, stride = stride_trj, probability = 1.0)

    print(f'{len(structures):} structures kept')
    print()

    return structures


# settings - n2p2 constructor
kwargs_model = dict(
    elements = ['Mg',],
    n = 8,
    fn_template = 'input.nn',
    n_tasks = 8,
    n_core_task = 16,
    remove_output = True
)

# structures to select from
structures = read_structures()

qbc = aml.QbC(
    structures = structures,
    cls_model = aml.N2P2,
    kwargs_model = kwargs_model,
    n_train_initial = 20,
    n_add = 20,
    n_epoch = 15,
    n_iterations = 15,
    n_candidate = 2001,
    fn_results = 'results.shelf',
    fn_restart = None
)

qbc.run()
