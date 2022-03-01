# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).  A nice improvement over GraphWavenet is presented by Shleifer et al. [paper](https://arxiv.org/abs/1912.07390) [code](https://github.com/sshleifer/Graph-WaveNet).



<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- see `requirements.txt`

- Run in virtual environment

# == EDITED ==

## Data Preparation

### Step1: Download temperature data from ORIGIN 

IMPROVED_full_max_temps_reduced.h5

### Step2: Process raw data

```
# Create data directories
mkdir -p data/MAX-TEMP_improved

# METR-LA
python generate_training_data.py --output_dir=data/MAX-TEMP_improved --traffic_df_filename=data/IMPROVED_full_max_temps_reduced.h5


```
## Train Commands

```
python train.py --gcn_bool --adjtype identity --addaptadj  --randomadj --data=data/MAX-TEMP_improved
```
