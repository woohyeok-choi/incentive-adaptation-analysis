# Incentive Adaptation Analysis


## Virutal Environment
```shell
conda env create -f env.yaml
conda activate incentive
```
Or, make your own conda virtual environment and install pacakges below:

```shell
conda install -c conda-forge r-base r-cairo r-tidyverse r-ggforce r-ggpubr r-lme4 r-psych r-showtext r-sjplot r-wrs2 r-effectsize r-car r-optimx r-extrafont jupyterlab ipywidgets rpy2 pandas numpy pytz altair pygeohash
```

## Directory
- [momab-simulation/notebook-momab.ipynb](momab-simulation/notebook-momab.ipynb)
  - The simulation results with the hypothetical user behaviors and incentive strategies

- [golden-time-analysis/golden-time.ipynb](golden-time-analysis/golden-time.ipynb)
  - The Golden Time data analysis codes


## Note
All notebooks are assumed to be executed in the Ubuntu environment (because **ray** is fully compatible with the Ubuntu).
If you do not have any Linux development environment, I strongly recommend to use WSL.

