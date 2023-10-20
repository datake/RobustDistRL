# Official implementation of 'Exploring the Training Robustness of Distributional Reinforcement Learning against Noisy State Observations' (ECML-PKDD 2023)

Please refer to [RobustDistRL](https://arxiv.org/abs/2109.08776) (ECML-PKDD 2023) to look into the details of our paper (we will further update the final version of our paper shortly).


### Step 1: install OpenAI baselines
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```
> Please refer to https://github.com/openai/baselines.

### Step 2: Update the environment

```
pip install -r requirement.txt
gym 0.12.1  
pytorch 1.7.0  
atari-py
cuda 11.0
```

### Step 3: traing models under noisy environments

We compare the learning curves of DQN, QRDQN and C51 on two Atari games (breakout and Qbert) and classical control environments (mountaincar and cartpole) across random and adversarial state noises. We take DQN for examples in the following code(change DQN to QRDQN / C51 to test distributional RL algorithms):

#### (1) For Normal Training (--train_env 0: no noises):
```
python main.py --game cartpole --iter 200000 --method DQN --train_env 0 --random 0.00 --epsilon 0.00 --seed 1 --gpu 0
python main.py --game mountaincar --iter 100000 --method DQN --train_env 0 --random 0.00 --epsilon 0.00 --seed 1 --gpu 0
python main.py --game breakout --iter 10000000 --method DQN --train_env 0 --random 0.00 --epsilon 0.0 --seed 1 --gpu 0
python main.py --game qbert --iter 10000000 --method DQN --train_env 0 --random 0.00 --epsilon 0.00 --seed 1 --gpu 0
```
#### (2) For Training under Random Noise (--train_env 1: random noise, --random X for difference perturbation strength):
```
python main.py --game cartpole --iter 200000 --method DQN --train_env 1 --random 0.05 --epsilon 0.00 --seed 1 --gpu 0
python main.py --game mountaincar --iter 100000 --method DQN --train_env 1 --random 0.0125 --epsilon 0.00 --seed 1 --gpu 0
python main.py --game breakout --iter 10000000 --method DQN --train_env 1 --random 0.01 --epsilon 0.0 --seed 1 --gpu 0
python main.py --game qbert --iter 10000000 --method DQN --train_env 1 --random 0.05 --epsilon 0.00 --seed 1 --gpu 0
```

#### (3) For Training under Adversarial Noise (--train_env 2: adversarial noise, --epsilon X for difference perturbation strength):
```
python main.py --game cartpole --iter 200000 --method DQN --train_env 2 --random 0.00 --epsilon 0.05 --seed 1 --gpu 0
python main.py --game mountaincar --iter 100000 --method DQN --train_env 2 --random 0.00 --epsilon 0.01 --seed 1 --gpu 0
python main.py --game breakout --iter 10000000 --method DQN --train_env 2 --random 0.00 --epsilon 0.0005 --seed 1 --gpu 0
python main.py --game qbert --iter 10000000 --method DQN --train_env 2 --random 0.00 --epsilon 0.005 --seed 1 --gpu 0
```

**Note**: Hyper-parameters can be modified with different arguments in different environments. Please refer to the paper for more details. We run Atari games for three times and cartpole, mountaincar for 200 times. It is suggested to run our code on the **breakout** game for the first time use to get more sense of the robustness advantage of distributional RL algorithms.

## Contact

If you have any questions or want to report a bug, it is more suggested to open an issue here. Alternatively, you can also send an email to ksun6@ualberta.

## Reference
Please cite our paper if you use this code in your own work (we will update shortly once the proceeding of the paper is finished):
```
@inproceedings{sun2023exploring,
  title={Exploring the training robustness of distributional reinforcement learning against noisy state observations},
  author={Sun, Ke and Zhao, Yingnan and Jui, Shangling and Kong, Linglong},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={36--51},
  year={2023},
  organization={Springer}
}
```

## Acknowledgement
We appreciate the following github repos a lot for their valuable code base:

https://github.com/ShangtongZhang/DeepRL
