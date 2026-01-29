# EPINR

This repository contains research and project code for the EPINR susceptibility distortion correction method.

The EPINR paper may be found at <https://openreview.net/forum?id=xrISwrfA0Y>

## Environment Setup

### Quickstart

To quickly recreate the development environment, install the anaconda packages found in
`mrinr.txt` and the pypi packages found in `requirements.txt`. For example:

```bash
# Clone the repository
git clone git@github.com:TylerSpears/epinr.git
cd mr-inr
# Make sure to install mamba in your root anaconda env for package installation.
# Explicit anaconda packages with versions and platform specs. Only works on the same
# platform as development.
mamba create --name mrinr --file mrinr.txt
# Move to the new environment.
conda activate mrinr
# Install pip packages, try to constrain by the anaconda package versions, even if pip
# does not detect some of them.
pip install --requirement requirements.txt --constraint pip_constraints.txt
# Install as a local editable package.
pip install -e .
```

### Detailed Installation Notes

If the previous commands fail to install the environment (which they likely will), then
the following notes should be sufficient to recreate the environment.

* All package versions are recorded and kept up-to-date in the `environment/` directory. If you encounter issues, check these files for the exact versions used in this code. Further instructions are found in the directory's `README.md`.
* All packages were installed and used on a Linux x86-64 system with Nvida GPUs. Using this code on Windows or Mac OSX is not supported.
* This environment is managed by [`mamba`](https://github.com/mamba-org/mamba), which wraps `anaconda`. `mamba` requires that no packages come from the `defaults` anaconda channel (see <https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html#using-the-defaults-channels> for details). All anaconda packages come from the following anaconda channels:
  * `conda-forge`
  * `pytorch`
  * `nvidia`
  * `simpleitk`
  * `mrtrix3`
  * `nodefaults` (simply excludes the `defaults` channel)
* Various packages conflict between `anaconda` and pypi, and there's no great way to resolve this problem. Generally, you should install `anaconda` packages first, then `pip install` packages from pypi, handling conflicts on a case-by-case basis. Just make sure that pip does not override `pytorch` packages that make use of the GPU.
* The `antspyx` package is not versioned because old versions of this package get deleted from pypi. See <https://github.com/ANTsX/ANTsPy#note-old-pip-wheels-will-be-deleted>

To install this repository as a python package, install directly from github:

```bash
pip install git+ssh://git@github.com/TylerSpears/epinr.git
```

To install an editable version for development:

```bash
pip install -e .
```

Note that the dependency package versions are *not* pinned for this repository, so you will need
to manually install required packages (pytorch, numpy, jaxlib, etc.).

## Directory Layout

This repository has the following top-level directory layout:

```bash
./ ## Project root
├── README.md
├── notebooks/ ## Notebooks for training, testing, and results analysis
├── scripts/ ## Scripts for data preprocessing, testing models, and other utilities
├── environment/ ## Detailed specs for package versions
├── mrinr/ ## Python package containing data loading/processing, metrics, etc.
├── tests/ ## Unit test scripts run by `pytest`
├── mrinr.txt ## Anaconda environment package specs
├── requirements.txt ## Pypi-installed package specs
└── pip_constraints.txt ## Constraints on pypi packages to help (slightly) differences between conda and pip
```

## Developers

### Installing Packages

When installing a new python package, always use [`mamba`](https://github.com/mamba-org/mamba)
for installation; this will save you so much time and effort. For example:

```bash
# conda install numpy
# replaced by
mamba install numpy
```

If a package is not available on the anaconda channels, or a package must be built from
a git repository, then use `pip`:

```bash
pip install ipyvolume
```

### Environment Variables

Project-specific environment variables may stored in a `.env` file. For convenience, you
may want to set up [`direnv`](<https://direnv.net/>) for automatic variable loading. Your
`.env` file should be specific to your system and may contain sensitive data or keys, so
it is explicitly *not* version-controlled.

See the `.env.template` and `.envrc.template` files for all env vars and example values, and for a starting
point for your own setup.

### git Config

The provided `.gitmessage` commit template can be configured for this repository with:

```bash
git config --local commit.template .gitmessage
```

### pre-commit Hooks

This repository relies on [`pre-commit`](<https://pre-commit.com/>) to run basic cleanup
and linting utilities before a commit can be made. Hooks are defined in the
`.pre-commit-config.yaml` file. To set up pre-commit hooks:

```bash
# If pre-commit is not already in your conda environment
mamba install -c conda-forge pre-commit
pre-commit install
# (Optional) run against all files
pre-commit run --all-files
```

### git Filters

The [`nbstripout`](<https://github.com/kynan/nbstripout>) application is set up as
a git repository filter that strips jupyter/ipython notebooks (*.ipynb files) of output
and metadata. Install `nbstripout` with:

```bash
# If not installed in your conda env already
mamba install -c conda-forge nbstripout
# If you have already cloned this repository, nbstripout is already in the .gitattributes.
# Otherwise, this will add the necessary lines:
nbstripout --install --attributes .gitattributes
```

You may selectively keep cell outputs in jupyter itself by tagging a cell with the
`keep_output` tag. See <https://github.com/kynan/nbstripout#keeping-some-output> for
details.
