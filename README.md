# CODAC
Instructions for installing the software and running reproducing experiments for CODAC

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
The following commands reproduce the experiments, note that the scripts are designed to be run on high performance cluster.
The results are created in `experiments/results`.

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

# Figure 4 and Table 1
bash start_experiments_guess_k.sh

# Figure 5
bash start_experiments_train_test_codac.sh

# Figure 6
bash start_experiments_abl_loss.sh

# Figure 7
bash start_experiments_abl_sampling.sh

# Figure 8
bash start_experiments_abl_deep_clusterers.sh
```

## Creating figures

The Figures in the article are generated with `create_clust_figs.py`, the script exptects the first parameter(s) to be directories for results, `-o` specifies the output directory, `-t <str>` is a style parameter for spefifying correct legend names, and `-l <str>` defines to which figure to add a legend to (default is `all`). Below is an example command for creating Figure 2.
```bash
# in experiments
python create_clust_figs.py results/aclust_results/aclust_compare_deep results/aclust_results/aclust_compare_shallow results/aclust_results/aclust_compare_a3s_shallow results/aclust_results/aclust_compare_a3s_deep results/aclust_results/aclust_compare_ffqs_shallow results/aclust_results/aclust_compare_ffqs_deep -o figures/aclust_new -t compare -l bloodmnist
```
