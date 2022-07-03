import pathlib

import numpy as np
import healpy as hp
from mpi4py import MPI

import pymods.io_mod


def get_mean_map(name, output_dir, n_samples, map_input, map_fnames, 
        mpi_comm, mpi_rank, component="CMB"):
    """
    Method for calculating and saving the mean map for a given component. 
    If the map already exists on disk, then it will be retrieved instead.

    Returns 
    -------
    map_mu mean map
    """

    plot_fname = f"{name.lower()}_{component.lower()}_meanmap_s{str(n_samples).zfill(4)}.fits"
    if not pathlib.Path(output_dir.joinpath(plot_fname)).exists():
        if mpi_rank == 0:
            print("---------------------------")
            print(f"Calculating Mean {component} map")
            print("---------------------------")
        
        # Creating an empty map
        map_mu = np.zeros_like(map_input)

        for i in range(len(map_fnames)):
            map_i  = pymods.io_mod.load_map(map_fnames[i]) 
            map_mu += map_i

        # Summing up the map (only master process will have the values)
        map_mu = mpi_comm.reduce(map_mu, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            map_mu = map_mu / n_samples
            print("[Done]")
            print("---------------------------")
            print(f"Saving Mean {component} map")
            print("---------------------------")
            hp.write_map(output_dir.joinpath(plot_fname).resolve(), map_mu)
    else:
        if mpi_rank == 0:
            print("---------------------------")
            print(f"Retrieving Mean {component} map")
            print("---------------------------")
            map_mu = hp.read_map(output_dir.joinpath(plot_fname), field=(0,1,2))
        else:
            map_mu = np.zeros_like(map_input)

    # Bacause we reduced an array (or created the one with zeros), 
    # the actual data exists only on Master process, Thus, we need 
    # to broadcast it to get further calculations done.
    map_mu = mpi_comm.bcast(map_mu, root=0)

    return map_mu


def get_rms_map(name, output_dir, n_samples, map_input, map_fnames, map_mu, 
        mpi_comm, mpi_rank, component="CMB"):
    """
    Method for calculating and saving the rms map for a given component. 
    If the map already exists on disk, then it will be retrieved instead.

    Returns 
    -------
    map_rms rms map
    """

    plot_fname = f"{name.lower()}_{component.lower()}_rmsmap_s{str(n_samples).zfill(4)}.fits"
    if not pathlib.Path(output_dir.joinpath(plot_fname)).exists():
        if mpi_rank == 0:
            print("---------------------------")
            print(f"Calculating RMS {component} map")
            print("---------------------------")

        # Creating an empty map
        map_rms = np.zeros_like(map_input)

        for i in range(len(map_fnames)):
            map_i  = pymods.io_mod.load_map(map_fnames[i]) 
            map_rms += (map_i - map_mu)**2

        # Summing up the map (only master process will have the values)
        map_rms = mpi_comm.reduce(map_rms, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            map_rms = np.sqrt(map_rms / (n_samples-1))
            print("[Done]")
            print("---------------------------")
            print(f"Saving RMS {component} map")
            print("---------------------------")
            hp.write_map(output_dir.joinpath(plot_fname).resolve(), map_rms)
    else:
        if mpi_rank == 0:
            print("---------------------------")
            print(f"Retrieving RMS {component} map")
            print("---------------------------")
            map_rms = hp.read_map(output_dir.joinpath(plot_fname), field=(0,1,2))
        else:
            map_rms = np.zeros_like(map_input)

    # Bacause we reduced an array (or created the one with zeros), 
    # the actual data exists only on Master process, Thus, we need 
    # to broadcast it to get further calculations done.
    map_rms = mpi_comm.bcast(map_rms, root=0)

    return map_rms

def get_delta_map(name, output_dir, n_samples, map_input, map_fnames, map_mu, 
        map_rms, mpi_comm, mpi_rank, component="CMB"):
    """
    Method for calculating (Mean - Input) / STD
    """

    # Creating an empty map
    map_delta = np.zeros_like(map_input)
    
    # (Mean - Input) / STD
    time1 = MPI.Wtime()
    plot_fname = f"{name.lower()}_{component.lower()}_delta_s{str(n_samples).zfill(4)}.fits"
    if not pathlib.Path(output_dir.joinpath(plot_fname)).exists():
        if mpi_rank == 0:
            print("---------------------------")
            print(f"Starting calculation of (Mean-Input)/RMS {component} map")
            print("---------------------------")
            map_delta = (map_mu - map_input) / map_rms
            print("[Done]")
            print("---------------------------")
            print(f"Saving (Mean-Input)/RMS {component}")
            print("---------------------------")
            hp.write_map(output_dir.joinpath(plot_fname).resolve(), map_delta)
        #else:
        #    map_delta = []
        #map_delta = mpi_comm.bcast(map_delta, root=0)
    else:
        if mpi_rank == 0:
            print("---------------------------")
            print(f"Retrieving (Mean-Input)/RMS {component}")
            print("---------------------------")
            map_delta = hp.read_map(output_dir.joinpath(plot_fname), field=(0,1,2))
        #else:
        #    map_delta = []
        #map_delta = mpi_comm.bcast(map_delta, root=0)
    time2 = MPI.Wtime()
    if mpi_rank == 0:
        print(f"Finished in {time2 - time1:0.2f}s")

    map_delta = mpi_comm.bcast(map_delta, root=0)

    return map_delta
