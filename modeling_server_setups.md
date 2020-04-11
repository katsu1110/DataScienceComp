(GCP)
General introduction to run a jupyter notebook in GCP
https://tudip.com/blog-post/run-jupyter-notebook-on-google-cloud-platform/


(GENERAL)
### Cautions
- better to choose CentOS (Ubuntu may be a bit troublesome)

### linux related
sudo apt install unzip

### install anaconda
wget https://repo.continuum.io/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh

### install must packages
conda install -c conda-forge lightgbm
conda install -c conda-forge xgboost
conda install -c conda-forge catboost
conda install -c conda-forge keras
conda install -c anaconda pytables
pip install pyarrow
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge optuna
conda install -c conda-forge matplotlib-venn
pip install waterfallcharts
conda install -c plotly plotly
conda install -c conda-forge missingno
conda install -c anaconda statsmodels
conda install -c anaconda astropy

(Troubleshooting)
### package not in path
add the following in .bashrc
export PATH=$PATH:/home/kakawagu/.local/bin
