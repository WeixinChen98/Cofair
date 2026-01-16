

# Cofair

The official implementation of HCRS@WWW2026 paper "Post-Training Fairness Control: A Single-Train Framework for Dynamic Fairness in Recommendation".

## Datasets
The preprocessed Movielens-1M and Lastfm-360K dataset are already provided in the `./data/ml1m` and `./data/lastfm` folders, respectively.

## Environments

The experimental environment is Python 3.10.11. We can first create and activate a new [Anaconda](https://www.anaconda.com/) environment for Python 3.10.11:
```
> conda create -n Cofair python=3.10.11
> conda activate Cofair
```

Then install all the required packages by using the command:
```
> pip install -r ./requirements.txt
```

## Usage
For example, for the MovieLens dataset, we can run the code of the assembly of Cofair fairness method on BPR recommendation model by running this command:
```
> cd ./src/
> nohup python -u main.py --fairness_model DSPT1 --recommendation_model BPR --dataset ml1m  --gpu_id 1 > ../results/ml1m/BPR_Cofair.log 2>&1 &
```

## Citation
If you find our work helpful, please consider citing our paper:
```bibtex
@inproceedings{chen2026posttraining,
  title     = {Post-Training Fairness Control: A Single-Train Framework for Dynamic Fairness in Recommendation},
  author    = {Chen, Weixin and Chen, Li and Zhao, Yuhan},
  year      = 2026,
  booktitle = {Companion Proceedings of the ACM Web Conference}
}
```

