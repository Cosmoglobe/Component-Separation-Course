"""
MPI implementation to speed things up

Installed with OpenMPI 4.1.4
"""
# Standard library modules
import os
import sys 
import pathlib
import json
from glob import glob
# Third-party modules
import numpy as np
import healpy as hp
import cosmoglobe
import matplotlib.pyplot as plt
from tqdm import tqdm 
from mpi4py import MPI
# Custom modules
import pymods
#import mpi_mod
#import io_mod
#import statistics_mod
#import plot_mod



def load_data(data_dir, chains_input_dir, chains_output_dir, sample_start, sample_end, 
        name, components, comp, mpi_comm, mpi_rank):
    """
    Methods for retrieveing Commander3 data from disk

    Return
    ------
    data_input 
    data_output
    """

    if mpi_rank == 0:
        print("Retrieving data")
        #print(components[comp][2])

        if (comp == "CMB" or comp == "Ncorr"):
            # CMB map is located inside data, while Ncorr is inside chains_sim_input
            if (comp == "CMB"):
                data_input = data_dir.joinpath(components[comp][1])
            elif(comp == "Ncorr"):
                data_input = chains_input_dir.joinpath(components[comp][1])
            # Working with input map only on one core
            data_input    = hp.read_map(data_input, field=(0,1,2))
            data_input[0] = hp.remove_dipole(data_input[0])
            data_input    = hp.ud_grade(data_input, 512)
            #data_output = []
            data_output = glob(f'{chains_output_dir}/{components[comp][2]}')
            #print(f"[Debug Msg] Total amount of {comp} samples on disk is {len(data_fnames)}")
            data_output.sort()
            data_output = data_output[sample_start:sample_end]

        #elif (comp == "Sigma0"):
        #    # Working with input
        #    data_input = f"{data_dir}/{components[comp][1]}"
        #    #print(data_input)
        #    idx = data_input.find("h5:")
        #    #print(idx)
        #    sample_input = int(data_input[idx+3])
        #    #print(sample)
        #    data_input = data_input[:idx+2]
        #    #print(data_input)
        #    #exit()
        #    #data_input = cosmoglobe.Chain(f"{data_dir}/chain_init_BP10.5.h5",
        #    #        samples=range(sample, sample+1)) 
        #    chain_input = cosmoglobe.Chain(data_input)
        #    data_input  = chain_input.get("tod/030/xi_n", 
        #            samples=range(sample_input, sample_input+1)) 
        #    exit()

    else:
        data_output = []
        data_input  = []

    if (comp == "CMB" or comp == "Ncorr"):
        # Broadcasting an entire array and reducing its size 
        # on each core  to conserve the memory
        data_output = mpi_comm.bcast(data_output, root=0)
        data_output = data_output[chunk_start:chunk_end]

    return data_input, data_output


def main():
    """
    Main Method
    """
    return


if __name__ == "__main__":

    start_time = MPI.Wtime()
    # Initialising MPI Communicators
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    if world_rank == 0:
        print(f"Program starts after {start_time:0.2f}s")

    # Root Path
    rootpath    = pathlib.Path(__file__).parent.resolve()
    data_dir    = rootpath.joinpath("data").resolve()
    # Creating Output directories
    output_dir  = rootpath.joinpath("output").resolve()
    plots_dir   = rootpath.joinpath("output", "plots").resolve()
    fitsout_dir = rootpath.joinpath("output", "fitsout").resolve() 

    if world_rank == 0:
        if not pathlib.Path.is_dir(output_dir):
            pathlib.Path.mkdir(output_dir)

        if not pathlib.Path.is_dir(plots_dir):
            pathlib.Path.mkdir(plots_dir)

        if not pathlib.Path.is_dir(fitsout_dir):
            pathlib.Path.mkdir(fitsout_dir)

    # Reading-in the parameter file 
    parameters = "params.json"
    with open(parameters, 'r') as myfile:
        data = myfile.read()
    parameters = json.loads(data)


    #print(parameters)
    for name, params in parameters.items():
        # Getting samples to work with
        sample_start = params["samples"]["start"]
        sample_end   = params["samples"]["end"]
        n_samples    = sample_end - sample_start
        # Splitting workload into chunks
        chunk_start, chunk_end, workloads = pymods.mpi_mod.split_workload(n_samples, 
                world_rank, world_size)
        print(f"World Size: {world_size}. Rank: {world_rank}. Samples: {chunk_start} -- {chunk_end}")
        # Current Path
        cpath             = rootpath.joinpath(f"{name}").resolve()
        chains_input_dir  = cpath.joinpath(params["datadirs"]["input"]).resolve()
        chains_output_dir = cpath.joinpath(params["datadirs"]["output"]).resolve()
        # 
        components = params["components"]
        if world_rank == 0:
            print(f"[Debug Msg]: {components}")

        # Calculating Statistics for every component
        for comp in components.keys():
            #print(components[comp][0])
            # Including only components with tag "yes" 
            if components[comp][0] == "y":
                if world_rank == 0:
                    print(f"Working with {comp}")

                # TODO: Add retrieval of data for Gain, Sigma0 etc.
                time1 = MPI.Wtime()
                data_input, data_output = load_data(data_dir, chains_input_dir, 
                        chains_output_dir, sample_start, sample_end, name, 
                        components, comp, world_comm, world_rank)
                time2 = MPI.Wtime()
                if world_rank == 0:
                    print(f"Finished in {time2 - time1:0.2f}s")

                # TODO: Rewrite this routine to calculate mean of Sigma0, Gain etc.
                # Mean
                time1 = MPI.Wtime()

                map_mu = pymods.statistics_mod.get_mean_map(name, fitsout_dir, n_samples, 
                        data_input, data_output, world_comm, world_rank, comp)
                # Debug
                if world_rank == 0:
                    print(f"[Debug Msg] Rank {world_rank}: Mean shape is {np.shape(map_mu)}, pixel 1 is {map_mu[1][1]}") 
                time2 = MPI.Wtime()
                if world_rank == 0:
                    print(f"Finished in {time2 - time1:0.2f}s")
                
                # TODO: Rewrite this routine to calculate STD of Sigma0, Gain etc.
                # RMS (STD)
                time1 = MPI.Wtime()

                map_rms = pymods.statistics_mod.get_rms_map(name, fitsout_dir, n_samples, 
                        data_input, data_output, map_mu, world_comm, world_rank, comp)
                # Debug 
                if world_rank == 0:
                    print(f"[Debug Msg] Rank {world_rank}: RMS shape is {np.shape(map_rms)}, pixel 1 is {map_rms[1][1]}") 
                time2 = MPI.Wtime()
                if world_rank == 0:
                    print(f"Finished in {time2 - time1:0.2f}s")

                map_delta = pymods.statistics_mod.get_delta_map(name, fitsout_dir, n_samples, 
                        data_input, data_output, map_mu, map_rms, world_comm, world_rank, comp)

                # Plotting maps and histograms
                time1 = MPI.Wtime()
                if world_rank == 0:
                    plot_fname = f"{name.lower()}_{comp.lower()}_map_s{str(n_samples).zfill(4)}.png"
                    pymods.plot_mod.plot_component_maps(plots_dir, plot_fname, 
                            data_input, map_mu, map_rms, map_delta, comp)
                    plot_fname = f"{name.lower()}_{comp.lower()}_hist_s{str(n_samples).zfill(4)}.png"
                    pymods.plot_mod.plot_component_hists(plots_dir, plot_fname, map_delta)
                time2 = MPI.Wtime()
                if world_rank == 0:
                    print(f"Finished in {time2 - time1:0.2f}s")

                world_comm.Barrier()

            else:
                print(f"Component {comp} is ignored")

    end_time   = MPI.Wtime()
    if world_rank == 0:
        print(f"Program finished in {(end_time - start_time):0.2f}s")

    # -------------------------------------


    #for name, params in parameters.items():
    #    sample_start = params[0]
    #    sample_end   = params[1]
    #    n_samples    = sample_end - sample_start

    #    chunk_start, chunk_end, workloads = pymods.mpi_mod.split_workload(n_samples, 
    #            world_rank, world_size)
    #    print(f"World Size: {world_size}. Rank: {world_rank}. Samples: {chunk_start} -- {chunk_end}")

    #    # Current Path
    #    cpath      = rootpath.joinpath(f"{name}").resolve()
    #    data_dir   = cpath.joinpath('chains_sim_input').resolve()
    #    chains_dir = cpath.joinpath('chains_proc').resolve()

    #    components = params[2]
    #    if world_rank == 0:
    #        print(f"[Debug Msg]: {components}")

    #    for comp in components.keys():
    #        # Including only components with tag "yes" 
    #        if components[comp][0] == "y":
    #            if world_rank == 0:
    #                print(f"Working with {comp}")

    #            time1 = MPI.Wtime()
    #            if world_rank == 0:
    #                print("Retrieving data")

    #                if (comp == "CMB" or comp == "Ncorr"):
    #                    data_fnames = glob(f'{chains_dir}/{components[comp][2]}')
    #                    print(f"[Debug Msg] Total amount of {comp} samples on disk is {len(data_fnames)}")
    #                    data_fnames.sort()
    #                    data_fnames = data_fnames[sample_start:sample_end]

    #                    # Working with input map only on one core
    #                    data_input    = hp.read_map(data_dir.joinpath(components[comp][1]), field=(0,1,2))
    #                    data_input[0] = hp.remove_dipole(data_input[0])
    #                    data_input    = hp.ud_grade(data_input, 512)
    #                elif (comp == "Sigma0"):
    #                    # Working with input
    #                    data_input = f"{data_dir}/{components[comp][1]}"
    #                    #print(data_input)
    #                    idx = data_input.find("h5:")
    #                    #print(idx)
    #                    sample_input = int(data_input[idx+3])
    #                    #print(sample)
    #                    data_input = data_input[:idx+2]
    #                    #print(data_input)
    #                    #exit()
    #                    #data_input = cosmoglobe.Chain(f"{data_dir}/chain_init_BP10.5.h5",
    #                    #        samples=range(sample, sample+1)) 
    #                    chain_input = cosmoglobe.Chain(data_input)
    #                    data_input  = chain_input.get("tod/030/xi_n", 
    #                            samples=range(sample_input, sample_input+1)) 
    #                    exit()

    #            else:
    #                data_fnames = []
    #                data_input  = []

    #            # Broadcasting an entire array and reducing its size 
    #            # on each core  to conserve the memory
    #            data_fnames = world_comm.bcast(data_fnames, root=0)
    #            data_fnames = data_fnames[chunk_start:chunk_end]

    #            data_input  = world_comm.bcast(data_input, root=0)
    #            time2 = MPI.Wtime()
    #            if world_rank == 0:
    #                print(f"Finished in {time2 - time1:0.2f}s")

    #            # Mean
    #            time1 = MPI.Wtime()

    #            map_mu = pymods.statistics_mod.get_mean_map(name, fitsout_dir, n_samples, 
    #                    data_input, data_fnames, world_comm, world_rank, comp)
    #            # Debug
    #            if world_rank == 0:
    #                print(f"[Debug Msg] Rank {world_rank}: Mean shape is {np.shape(map_mu)}, pixel 1 is {map_mu[1][1]}") 
    #            time2 = MPI.Wtime()
    #            if world_rank == 0:
    #                print(f"Finished in {time2 - time1:0.2f}s")

    #            # RMS (STD)
    #            time1 = MPI.Wtime()

    #            map_rms = pymods.statistics_mod.get_rms_map(name, fitsout_dir, n_samples, 
    #                    data_input, data_fnames, map_mu, world_comm, world_rank, comp)
    #            # Debug 
    #            if world_rank == 0:
    #                print(f"[Debug Msg] Rank {world_rank}: RMS shape is {np.shape(map_rms)}, pixel 1 is {map_rms[1][1]}") 
    #            time2 = MPI.Wtime()
    #            if world_rank == 0:
    #                print(f"Finished in {time2 - time1:0.2f}s")

    #            map_delta = pymods.statistics_mod.get_delta_map(name, fitsout_dir, n_samples, 
    #                    data_input, data_fnames, map_mu, map_rms, world_comm, world_rank, comp)

    #            # Plotting maps and histograms
    #            time1 = MPI.Wtime()
    #            if world_rank == 0:
    #                plot_fname = f"{name.lower()}_{comp.lower()}_map_s{str(n_samples).zfill(4)}.png"
    #                pymods.plot_mod.plot_component_maps(plots_dir, plot_fname, 
    #                        data_input, map_mu, map_rms, map_delta, comp)
    #                plot_fname = f"{name.lower()}_{comp.lower()}_hist_s{str(n_samples).zfill(4)}.png"
    #                pymods.plot_mod.plot_component_hists(plots_dir, plot_fname, map_delta)
    #            time2 = MPI.Wtime()
    #            if world_rank == 0:
    #                print(f"Finished in {time2 - time1:0.2f}s")

    #            world_comm.Barrier()

    #        else:
    #            print(f"Component {comp} is ignored")

    #end_time   = MPI.Wtime()
    #if world_rank == 0:
    #    print(f"Program finished in {(end_time - start_time):0.2f}s")



    

    # Specifying the burn-in and the amount of samples we want to use
    #sample_start = 100
    #sample_end   = 3100
    #n_samples    = sample_end - sample_start

    #chunk_start, chunk_end, workloads = mpi_mod.split_workload(n_samples, world_rank, world_size)
    #print(f"World Size: {world_size}. Rank: {world_rank}. Samples: {chunk_start} -- {chunk_end}")

    # Current Path
    #cpath      = pathlib.Path(".")
    ##data_dir   = cpath.joinpath('data').resolve()
    #data_dir   = cpath.joinpath('chains_sim_input').resolve()
    #chains_dir = cpath.joinpath('chains_proc').resolve()
    #plots_dir  = cpath.joinpath("plots").resolve()
    #output_dir = cpath.joinpath("output").resolve()

    #if world_rank == 0:
    #    # Creating directories (if needed) 
    #    if not pathlib.Path.is_dir(plots_dir):
    #        pathlib.Path.mkdir(plots_dir)
    #    if not pathlib.Path.is_dir(output_dir):
    #        pathlib.Path.mkdir(output_dir)

    #components = {"CMB": "cmb_c0001_k*.fits", "Ncorr": "tod_030_ncorr_c0001_k*.fits"}
    #components = {"CMB": "my_cmb_map.fits", "Ncorr": "tod_030_ncorr_c0001_k000001.fits"}

    #for comp, comp_fnames in components.items():
    #    if world_rank == 0:
    #        print(f"Working with {comp}")
    #
    #    time1 = MPI.Wtime()
    #    if world_rank == 0:
    #        print("Retrieving data")

    #        #data_fnames = glob(f'{chains_dir}/cmb_c0001_k??????.fits')
    #        data_fnames = glob(f'{chains_dir}/{comp_fnames}')
    #        print(f"[Debug Msg] Total amount of {comp} samples on disk is {len(data_fnames)}")
    #        data_fnames.sort()
    #        data_fnames = data_fnames[sample_start:sample_end]

    #        # Working with input map only on one core
    #        data_input    = hp.read_map(data_dir.joinpath(components[comp]), field=(0,1,2))
    #        data_input[0] = hp.remove_dipole(data_input[0])
    #        data_input    = hp.ud_grade(data_input, 512)
    #    else:
    #        data_fnames = []
    #        data_input  = []

    #    #world_comm.Barrier()

    #    # Broadcasting an entire array and reducing its size 
    #    # on each core  to conserve the memory
    #    data_fnames = world_comm.bcast(data_fnames, root=0)
    #    data_fnames = data_fnames[chunk_start:chunk_end]

    #    data_input  = world_comm.bcast(data_input, root=0)
    #    time2 = MPI.Wtime()
    #    if world_rank == 0:
    #        print(f"Finished in {time2 - time1:0.2f}s")

    #    # Mean
    #    time1 = MPI.Wtime()

    #    map_mu = statistics_mod.get_mean_map(output_dir, n_samples, data_input, 
    #            data_fnames, world_comm, world_rank, comp)
    #    # Debug
    #    if world_rank == 0:
    #        print(f"[Debug Msg] Rank {world_rank}: Mean shape is {np.shape(map_mu)}, pixel 1 is {map_mu[1][1]}") 
    #    time2 = MPI.Wtime()
    #    if world_rank == 0:
    #        print(f"Finished in {time2 - time1:0.2f}s")
    #        
    #    # RMS (STD)
    #    time1 = MPI.Wtime()

    #    map_rms = statistics_mod.get_rms_map(output_dir, n_samples, data_input, 
    #            data_fnames, map_mu, world_comm, world_rank, comp)
    #    # Debug 
    #    if world_rank == 0:
    #        print(f"[Debug Msg] Rank {world_rank}: RMS shape is {np.shape(map_rms)}, pixel 1 is {map_rms[1][1]}") 
    #    time2 = MPI.Wtime()
    #    if world_rank == 0:
    #        print(f"Finished in {time2 - time1:0.2f}s")

    #    # (Mean - Input) / STD
    #    time1 = MPI.Wtime()
    #    plot_fname = f"{comp.lower()}_output_s{str(n_samples).zfill(4)}.fits"
    #    if not pathlib.Path(output_dir.joinpath(plot_fname)).exists():
    #        if world_rank == 0:
    #            print("---------------------------")
    #            print("Starting calculation of (Mean-Input)/RMS {comp} map")
    #            print("---------------------------")
    #            map_output = (map_mu - data_input) / map_rms
    #            print("[Done]")
    #            print("---------------------------")
    #            print("Saving (Mean-Input)/RMS {comp}")
    #            print("---------------------------")
    #            hp.write_map(output_dir.joinpath(plot_fname).resolve(), map_output)
    #        #else:
    #        #    map_output = []
    #        #map_output = world_comm.bcast(map_output, root=0)
    #    else:
    #        if world_rank == 0:
    #            print("---------------------------")
    #            print("Retrieving (Mean-Input)/RMS {comp}")
    #            print("---------------------------")
    #            map_output = hp.read_map(output_dir.joinpath(plot_fname), field=(0,1,2))
    #        #else:
    #        #    map_output = []
    #        #map_output = world_comm.bcast(map_output, root=0)
    #    time2 = MPI.Wtime()
    #    if world_rank == 0:
    #        print(f"Finished in {time2 - time1:0.2f}s")


    #    # Plotting maps and histograms
    #    time1 = MPI.Wtime()
    #    if world_rank == 0:
    #        plot_fname = f"{comp.lower()}_map_s{str(n_samples).zfill(4)}.png"
    #        plot_mod.plot_component_maps(plots_dir, plot_fname, data_input, map_mu, 
    #                map_rms, map_output, comp)
    #        plot_fname = f"{comp.lower()}_hist_s{str(n_samples).zfill(4)}.png"
    #        plot_mod.plot_component_hists(plots_dir, plot_fname, map_output)
    #    time2 = MPI.Wtime()
    #    if world_rank == 0:
    #        print(f"Finished in {time2 - time1:0.2f}s")

    #    world_comm.Barrier()
    # 
    #end_time   = MPI.Wtime()
    #if world_rank == 0:
    #    print(f"Program finished in {(end_time - start_time):0.2f}s")

        
