# ECMulate

The main ingredient is a surrogate model for an RC-loop. The model maps time-series of input current, resistance, and capacitance to the overall potential difference across the loop as a time series, starting from an initial voltage.

The duration of the time series is 100 seconds, with 128 equally spaced time-grid points. The package contains tools for inferring functional parameters in a 1-RC model using the surrogate binary. Since the surrogate is for an RC-loop only, it can be generalised to a n-RC Thevenin circuit.

## Requirements
The surrogate was compiled in a linux_x86_64 machine.

- `pytorch` 2.5.1
- `cuda` 12.1 (if used)
- `numpy`
- `interpolatorch` for `autograd` compatible batch interpolation tasks. Get it [here](http://github.com/gumrukcuoglu/interpolatorch)

Although actually optional, the following is required to import the library in this version (so for all intents and purposes, are required)
- `npchip` for PCHIP impementation to numpy. Get it [here](http://github.com/gumrukcuoglu/npchip)
- `matplotlib`
- `pandas`

The actual surrogate only requires `pytorch` (and `cuda` if used) to run, but for the additional features for inference, the requirement list is a bit large now.

To run the example notebook you will also need:
- `ipykernel` or `jupyter` (not included in requirements)

## Installation
### Binary installation:

1. Set-up a new conda environment:

        conda create -n ECMulate python=3.11 -c conda-forge
        conda activate ECMulate

3. Install package

~`pip install git+https://github.com/gumrukcuoglu/ECMulate`~ (This will become relevant if/when package goes public)

Download the repo as a zip file, locate the file, then:

        unzip ECMulate-master.zip
        cd ECMulate-master
        pip install .

2. Install dependencies

        pip install -r requirements.txt

## Usage

See the example notebook in `examples/inference.ipynb`


The surrogate is automatically loaded when the package is imported. If GPU exists, this is the default, otherwise it is loaded into the cpu. It can be loaded
If you wish to use the surrogate only, it can be called via:

`ECMulate.forward(X, Xin)`

### Inputs:

- `X`: A torch tensor of shape `(N, 3, 128)`. The tensor represents a batch of time series where:
  - `axis=1` corresponds to the following features:
    - Input current
    - Time-scale ($\tau = R \times C$, where $R$ is resistance and $C$ is capacitance)
    - Capacitance
  - All features are provided as time series of size `128` (along `axis=2`).
  - `N` represents the batch size and can be any integer, provided that memory constraints are not exceeded.

- `Xin`: A torch tensor of shape `(N, 1, 1)` representing the initial voltage of the system for each batch.

### Output:
- `y`: A torch tensor of shape `(N, 1, 128)`, corresponding to the output voltage as a time series for each input batch.

## Rescale/Descale

The module now contains two new functions:

- `ECMulate.input_rescaler(X, Xin)`, outputs rescaled `(X, Xin)`. Since the model was trained with standardised data, this rescaling ensures to map the physical inputs to the one compatible with the model.
- `ECMulate.output_descaler(y)`, outputs descaled `y`. This ensures the standardised output of the model is brought-back to physical scaling.


A proper documentation will be added soon.

### Units:

- Time is rescaled by 1000 seconds (i.e., `t=1` corresponds to 1000 seconds).
- Capacitance is rescaled by 1000 F.
- Resistance and voltage are in SI units.

## Version history:

- v0.1 : first working version. [18/10/2024]

    Added the `force_cpu` setting to `load_model()` to force loading to the cpu, regardless of the presence of a CUDA device.
- v0.2 : unplanned modification.

    Added the `input_rescaler` and `output_descaler` functions. [18/10/2024]
    Added an inference example in `examples/1RC_model_inference.ipynb` using data from Zülke et al. 2021. [22/10/2024]
    ## To do:
    Other models, implementing part of the inference within the package etc.

- v0.3 : Updated the surrogate to a newer (hopefully better) version, lots of hacking-protection and obfuscation on the actual model. Unfortunately this version is full of garbage. [02/05/2025]

- v0.4 : First clean and semi-private version, including data handling functions and inference functions. Comes with fully working inference notebook. [11/05/2025]
