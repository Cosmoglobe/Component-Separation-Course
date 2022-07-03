# PROJECT SUMMARY FOR TOD SIMULATION EXERCISE IN AST9240

This folder contains the materials for one of the AST9240 final projects.

## Steps-to-perform

0) Get access to a Linux-based machine with minimum of 128GB RAM

1) Download and compile the Commander `AST9240_v2` branch (see 
   [official documentation](git@github.com:Cosmoglobe/Component-Separation-Course.git) 
   for more details):

   a) cd your_preferred_src
   b) git clone https://github.com/Cosmoglobe/Commander.git
   c) cd Commander
   d) git checkout AST9240_v2
   e) mkdir build; cd build
   f) module load intel cmake
   g) For Intel compilation (recommended): cmake -DCMAKE_INSTALL_PREFIX=your_preferred_src/Commander/build -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DMPI_C_COMPILER=mpiicc -DMPI_CXX_COMPILER=mpiicpc -DMPI_Fortran_COMPILER=mpiifort -DFFTW_ENABLE_AVX2=OFF ..
   h) cmake --build . --target install -j 8
   i) Add the following lines to your .bashrc file:
   ```
      ---------------------------
      ulimit -Sd unlimited
      ulimit -Ss unlimited
      ulimit -c 0
      ulimit -n 2048
      
      export COMMANDER_PARAMS_DEFAULT=your_preferred_src/commander3/parameter_files/defaults
      export HEALPIX=/path/to/Healpix/root
      ```
      --------------------------
      For example, for Duncan, who used the `install_ita.sh` script, installed Commander in $HOME, so his .bashrc reads:
      ```
      export HEALPIX=$HOME/Commander/downloads/healpix
      export COMMANDER_PARAMS_DEFAULT="$HOME/Commander/commander3/parameter_files/defaults/"
   ```
2) Download data, parameter file and runscript:
   a) cd your_preferred_workdir
   b) wget http://tsih3.uio.no/www_cmb/hke/AST9240/AST9240_TODsim_project.tar
   c) tar xvf AST9240_TODsim_project.tar
   d) cd AST9240
   e) Change all pathnames in param_sim.txt, data/filelist_030_simulations.txt
      and data/filelist_030_data.txt
   f) Edit 'build' and numprocs in run_sim.sh

3) Produce simulations (TOD files will end up in data/LFI_030_sim):
   a) Create an ideal CMB (temperature+polarization) realization from
      the Planck 2018 best-fit power spectrum at Nside=1024, lmax=2000 
      and 14 arcmin FWHM resolution:
      http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt
      Add the following sky map to your simulated map, to account for
      the CMB dipole:
      http://tsih3.uio.no/www_cmb/hke/AST9240/map_cmb_dipole.fits
   b) Verify that the following parameters are set in params_sim.txt:
       - COMP_INPUT_AMP_MAP&&    = your_cmb_map.fits
       - ENABLE_TOD_SIMULATIONS  = .true.
       - SIMS_OUTPUT_DIRECTORY   = your_preferred_workdir/data/LFI_030_sim
       - BAND_TOD_FILELIST&&&    = filelist_030_data.txt
   c) ./run_sim.sh
   d) Move input sims to safe place: mv chains_sim chains_sim_input

4) Analyse simulations (output will end up in chains_sim):
   a) Change the following parameters in param_sim.sh:
       - ENABLE_TOD_SIMULATIONS  = .false.
       - BAND_TOD_FILELIST&&&    = filelist_030_sim.txt
   b) mkdir chains_sim
   c) ./run_sim.sh (let it run for at least 10 samples before proceeding;
      for the final results, we should have 1000. But it's useful to verify that
      the code is producing something useful before making the long run, so check
      often and early :-))

5) Compute posterior mean and RMS for the following quantities, using your
   favorite tools (healpy or whatever)
   a) CMB map (use cmb_c0001_k*.fits)
   b) Frequency sky map (use tod_030_map_c0001_k*.fits)
   c) Correlated noise (use tod_030_ncorr_c0001_k*.fits)
   d) Absolute gain for detector 27M (use chain_c0001.h5)
   e) Gain as a function of time for detector 27M (use chain HDF file)
   f) Chisq as a function of time for detector 27M (use chain HDF file)

6) Compute (posterior mean - input)/posterior rms for each quantity, and
   compare with an N(0,1) distribution
   
7) Deliver the end products to the owl server, either by getting an account or setting up some kind of ftp. The products you need to include are:
   i) Chain file, with the seed you used to generate the the map appended, e.g., with seeed 1345, your file should be chain_0001_s1345.h5.
   ii) Sky maps, tod_030_map_c0001_k*.fits.
   iii) Correlated noise maps, tod_030_ncorr_c0001_k*.fits
   iv) Output CMB maps, cmb_c0001_k??????.fits
   iv) Input maps, including your_cmb_map.fits (map used to generate sims), the correlated noise map from the simulation run (tod_030_ncorr_c001_k000001.fits in chains_sim_input, the directory where the first maps were generated.)
  
Some general clarifying points:
- The first run, which generates the simulated data, creates a correlated noise realization that is outputted, and writes the simulated timestreams to disk. Most of the fits maps come from a processing of the real LFI data, but the correlated noise map is exactly what is added to the simulated data.
- The second main run, which analyzes the simulated data from the first step, will output data in the new directory. In this directory, none of the maps should have any trace of the Galactic plane in the maps. If you see any of the Galaxy in your tod_030_map_c0001_k*.fits files, there has been an error, and you should make sure that the filelist given to Commander is actually pointing to the simulated data.
- The true instrumental parameters, including gain and noise parameters, are genearated from the first sample of chain_init_BP10.5.h5. 
- For the Commander run that we have set up, there should be ten files that are output.
-- chisq_c0001_k??????.fits - Nside = 16 map of the chi-squared statistic per pixel.
-- cmb_c0001_k??????.fits - Map of CMB estimate
-- instpar_c0001_k??????.dat - Table of sampled of map gain (unused parameter for us) and central bandpass shift, which can be sampled.
-- res_030_c0001_k??????.fits - Difference between the sky model evaluated at the Planck LFI 30 GHz bandpass and resolution, and the processed sky map. Difference taken in map space.
-- sigma_l_cmb_c0001_k??????.dat - Power spectrum of full-sky CMB map sample. Can be compared with the input C_l used to create the map in the first place.
-- tod_030_map_c0001_k??????.dat - Processed 030 map, after all instrumental parameters are sampled.
-- tod_030_ncorr_c0001_k??????.dat - Correlated noise timestream processed in the same way as the processed tod timestream, useful for identifying unmodeled systematics.
-- tod_030_res_c0001_k??????.dat - TOD - ncorr - model binned into a sky map. Shows components of the data that are not explained either by correlated noise or the data model.
-- tod_030_rms_c0001_k??????.fits - Standard deviation of the processed sky map, assuming a white noise contribution at every step.
-- tod_030_Smap_c0001_k??????.fits - Spurious component map, i.e., timestream component that corresponds to differences between timestreams due to different bandpasses. Used to evaluate whether to accept a bandpass shift proposal (smaller S => more likely to accept).

PS! The results from 6) and 7) will form the main results in the paper we will write.


How to install GCC+OpenMPI+CMake+Commander completely from source on Ubuntu:

It takes ~2,5 hrs.

1. Install GCC from source, I assume you are inside your `$HOME` directory:
```
$ mkdir local && cd local && mkdir src && cd src
$ wget https://mirror.koddos.net/gcc/releases/gcc-10.3.0/gcc-10.3.0.tar.gz && tar -xzvf gcc-10.3.0.tar.gz && cd gcc-10.3.0
$ ./contrib/download_prerequisites && mkdir build && cd build
$ ../configure --prefix=$HOME/local/gcc/10.3.0 --enable-languages=all --disable-multilib --enable-threads --enable-checking=release --with-system-zlib
$ make -j 8 # <= this step will take you 1-1,5 hrs
$ make install
```
and lastly modify your `.bashrc`  to point to an existing installation:
```
export PATH=$HOME/local/gcc/10.3.0/bin:$PATH
export MANPATH=$HOME/local/gcc/10.3.0/share/man:$MANPATH
export INFOPATH=$HOME/local/gcc/10.3.0/share/info:$INFOPATH
export LD_LIBRARY_PATH=$HOME/local/gcc/10.3.0/lib64:$LD_LIBRARY_PATH
```
and do not forget to source your `.bashrc`:
```
$ source .bashrc
```

2. Install an OpenMPI:
```
$ unset $FC
$ cd $HOME/local/src && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz && tar -xzvf openmpi-4.0.5.tar.gz && cd openmpi-4.0.5
$ ./configure --prefix=$HOME/local/gcc/10.3.0
$ make -j 8
$ make install
```

3. Install CMake:
```
$ cd $HOME/local/src && wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3.tar.gz && tar -xzvf cmake-3.21.3.tar.gz && cd cmake-3.21.3
$ ./bootstrap --prefix=$HOME/local/gcc/10.3.0 -- -DCMAKE_BUILD_TYPE:STRING=Release 
$ make -j 8
$ make install
```
Now, on each of these steps, YOU CHECK INSTALLED BINARIES WITH which `<binary name>`, e.g. `which gfortran`, or `which cmake`. You do the same to check their versions: `cmake --version`!

4. Finally, install Commander:
```
$ cd $HOME
$ git clone https://github.com/Cosmoglobe/Commander.git && cd Commander && git checkout AST9240_v2
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$HOME/Commander/build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx -DMPI_Fortran_COMPILER=mpif90 -DFFTW_ENABLE_AVX2=OFF -DUSE_SYSTEM_BLAS=OFF ..
$ cmake --build . --target install -j 8
```
and, finally, you modify your .bashrc  again to point to your Commander installation:
```
export PATH=$HOME/Commander/build/bin:$HOME/local/gcc/10.3.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/Commander/build/lib:$HOME/local/gcc/10.3.0/lib64:$LD_LIBRARY_PATH

export HEALPIX=$HOME/Commander/build/healpix
export COMMANDER_PARAMS_DEFAULT=$HOME/Commander/commander3/parameter_files/defaults
```
And that is it, you are done.
To check your installation you can try to do:
```
commander3 --version
```
It should give you version information.
If you have an MPI error, do not worry — it is expected, since some of you are on Cori or other big cluster machines which have a slurm job which cut mpi applications on login nodes.
