# CODAC
Instructions for installing the software and running reproducing experiments for CODAC (under review).

# Installing
```bash
# set up a conda enviroment
conda create -n codac python=3.13
conda activate codac
# install numpy if not installed yet (required by ClustPy)
pip install numpy
# clone the repository
git clone https://github.com/edahelsinki/codac.git
cd codac
# clone ClustPy
git clone https://github.com/collinleiber/ClustPy.git
cd ClustPy
# install the development version accordingly to the instructions in https://github.com/collinleiber/ClustPy
python setup.py install --prefix ~/.local
python setup.py build_ext --inplace

cd ..
# install additional requirements
pip install scikit-learn-extra, wandb, ucimlrepo, active-semi-supervised-clustering, nltk

# install project as a package
pip install -e .

# DONE
```

# Running experiments in HPC
Utilizing slurm for managing jobs, one can use existing scripts. The directory `experiments` contain the Python code for running experiment loading datasets and creating figures.

## Loading datasets

By running the following commands the data are dowloaded and cached automatically.
```bash
cd experiments
python data.py
```

## Running experiments
The following commands reproduce the experiments, note that the scripts are designed to be run on high performance cluster. One can see the `.sh` script for a reference how to run the experiments.
The results are created in `experiments/results`.

Reproducing Fig. 3:
```bash
# in experiments
cd scripts

# Figure 3
bash start_experiments_deep.sh
bash start_experiments_shallow.sh
bash start_experiments_a3s.sh
bash start_experiments_a3s_deep.sh
bash start_experiments_ffqs.sh
bash start_experiments_ffqs_deep.sh
```
