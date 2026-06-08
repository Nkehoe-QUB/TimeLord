# TimeLord
Python package to automate plotting of EPOCH or SMILEI data.

For EPOCH data, the `.sdf` files are converted to `.h5` files to reduce memory usage.

TimeLord can uses multiple CPU cores to process data conversion and plotting more efficiently and faster. The code will determine the amount of availbe CPU cores and memory and adjust accordingly for data conversion so that too much memory is not used.

Some assumptions are made:
- Laser wavelength is defined as: `lambda_las = ... * <nano or micron>` or `lambda_0 = ... * <nano or micron>`
- Distance from x_min boundary to target front is defined as: `xMin = ... * <nano or micron>`
- Pulse duration in intensity space of the laser is defined as: `Tau_I = ... * <femto or pico>` or `tau_fwhm_I = ... * <femto or pico>`
- Particle Binning diagnostics are names: `<species> density`, `<species> spectra`, `<species> <phase axis i.e. x-px, px-py> phase space`, `<species> angle`.

## Get TimeLord
Within the directory you want TimeLord to live, e.g `~/software`, run:
```
git clone https://github.com/Nkehoe-QUB/TimeLord
```
This will create a TimeLord directory.

## Create the conda environemnt and install dependencies
```
cd TimeLord
conda env create -f requiremnts.yml
```

## Install TimeLord into your environment
Within the TimeLord directory (`~/software/TimeLord`) run:
```
pip install -e .
```

## To use
In python run:
```
install timelord as tl

sim = tl.Open() # This will try open to open the current directory as a simualtion, alternatively run "sim = tl.Open(<PathToSim>)"

sim.Help() # This will show the available functions that can be ran and their inputs
```