#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import os, torch
from deep_rl import *
import argparse
import gym

parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('--game', default="breakout", type=str, choices=['cartpole', 'mountaincar', 'breakout', 'qbert'])
parser.add_argument('--method', default="QRDQN", type=str, help='DQN, C51')
parser.add_argument('--iter', default=1e7, type=int, help='number of iterations')
parser.add_argument('--train_env', default=0, type=int, help='0: normal training, 1: rand on current state, 2: adv on current state', choices=[0, 1, 2])
parser.add_argument('--PGD', default=3, type=int, help='iterations for PGD attack in adversarial state observations')
parser.add_argument('--random', default=0.0, type=float, help='standard deviations for the random noise while training')
parser.add_argument('--epsilon', default=0.0, type=float, help='epsilon in PGD, =0.0 if the random noise is applied')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--gpu', default=-1, type=int)
args = parser.parse_args()


if args.gpu != -1:
    os.environ["MKL_NUM_THREADS"] = '4'
    os.environ["NUMEXPR_NUM_THREADS"] = '4'
    os.environ["OMP_NUM_THREADS"] = '4'
    torch.set_num_threads(4)

#########################  DQN: images #######################
def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length)) # conv
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(args.iter) # number of iterations
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer() # image
    config.reward_normalizer = SignNormalizer() # sign
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.async_actor = True

    if args.train_env==0:
        config.logtxtname = args.method + str(args.iter)+'_seed' + str(args.seed)
    elif args.train_env==1:
        config.logtxtname = args.method + str(args.iter)+'_env'+str(args.train_env)+'_rand'+str(args.random) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_adv' + str(args.epsilon) + '_seed' + str(args.seed)
    config.epsilon = args.epsilon
    config.PGD = args.PGD
    config.adv_attack = False
    # training
    config.game_file = args.game
    config.random = args.random
    config.train_env = args.train_env
    config.method = args.method
    Agent = DQNAgent(config)
    run_steps(Agent, args)


#########################  DQN: feature vector #######################
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game) # read the game configurations
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001) # cartpole
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim)) # fully-connected network, state, 64, 64, action
    config.history_length = 1
    config.batch_size = 10
    config.discount = 0.99
    config.max_steps = int(args.iter)

    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length)

    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True # double Q learning
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.async_actor = False

    if args.train_env == 0:
        config.logtxtname = args.method + str(args.iter) + '_seed' + str(args.seed)
    elif args.train_env == 1:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_rand' + str(args.random) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_adv' + str(args.epsilon) + '_seed' + str(args.seed)
    config.epsilon = args.epsilon
    config.PGD = args.PGD
    config.adv_attack = False
    # training
    config.game_file = args.game
    config.random = args.random
    config.train_env = args.train_env
    config.method = args.method
    Agent = DQNAgent(config)
    run_steps(Agent, args)

#########################  QRDQN: images #######################
def quantile_regression_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, NatureConvBody()) # conv, different last FC layer
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.num_quantiles = 200 # N
    config.max_steps = int(args.iter)

    if args.train_env == 0:
        config.logtxtname = args.method + str(args.iter) + '_seed' + str(args.seed)
    elif args.train_env == 1:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_rand' + str(args.random) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_adv' + str(args.epsilon) + '_seed' + str(args.seed)
    config.epsilon = args.epsilon
    config.PGD = args.PGD
    config.adv_attack = True if args.test_env == 2 else False
    # training
    config.game_file = args.game
    config.random = args.random
    config.train_env = args.train_env
    config.method = args.method

    Agent = QuantileRegressionDQNAgent(config)
    run_steps(Agent, args)

#########################  QRDQN: feature vector #######################
def quantile_regression_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    if args.game == 'cartpole':
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001)
    else: # mountaincar
        print('mountain car change optimizer!')
        config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.0005)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, FCBody(config.state_dim)) # FC

    # config.batch_size = 10
    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size)
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.num_quantiles = 2 if args.game=='mountaincar' else 20
    config.gradient_clip = 5
    config.sgd_update_frequency = 4
    config.eval_interval = int(5e3)
    config.max_steps = int(args.iter)


    if args.train_env == 0:
        config.logtxtname = args.method + str(args.iter) + '_seed' + str(args.seed)
    elif args.train_env == 1:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_rand' + str(args.random) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_adv' + str(args.epsilon) + '_seed' + str(args.seed)
    config.epsilon = args.epsilon
    config.PGD = args.PGD
    config.adv_attack = True if args.test_env == 2 else False
    # training
    config.game_file = args.game
    config.random = args.random
    config.train_env = args.train_env
    config.method = args.method
    Agent = QuantileRegressionDQNAgent(config)
    run_steps(Agent, args)

#########################  C51: feature vector #######################
def categorical_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)

    config.batch_size = 10
    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size)
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 51
    config.gradient_clip = 5
    config.sgd_update_frequency = 4

    config.eval_interval = int(5e3)
    config.max_steps = int(args.iter)

    if args.train_env == 0:
        config.logtxtname = args.method + str(args.iter) + '_seed' + str(args.seed)
    elif args.train_env == 1:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_rand' + str(args.random) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_adv' + str(args.epsilon) + '_seed' + str(args.seed)
    config.epsilon = args.epsilon
    config.PGD = args.PGD
    config.adv_attack = True if args.test_env == 2 else False
    # training
    config.game_file = args.game
    config.random = args.random
    config.train_env = args.train_env
    config.method = args.method
    Agent = CategoricalDQNAgent(config)
    run_steps(Agent, args)


#########################  C51: images #######################
def categorical_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=True)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    config.max_steps = int(args.iter)

    if args.train_env == 0:
        config.logtxtname = args.method + str(args.iter) + '_seed' + str(args.seed)
    elif args.train_env in == 1:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_rand' + str(args.random) + '_seed' + str(args.seed)
    else:
        config.logtxtname = args.method + str(args.iter) + '_env' + str(args.train_env) + '_adv' + str(args.epsilon) + '_seed' + str(args.seed)
    config.epsilon = args.epsilon
    config.PGD = args.PGD
    config.adv_attack = True if args.test_env == 2 else False
    # training
    config.game_file = args.game
    config.random = args.random
    config.train_env = args.train_env
    config.method = args.method
    Agent = CategoricalDQNAgent(config)
    run_steps(Agent, args)


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    if args.gpu == -1:
        set_one_thread()
    random_seed()
    select_device(args.gpu) # -1: GPU, 0-8: GPU id

    # (0) save training results for each game
    if not os.path.exists('results'):
        os.makedirs('results')
    GAMES = ['cartpole', 'mountaincar', 'breakout', 'qbert']
    for i in GAMES:
        if not os.path.exists('results/'+i):
            os.makedirs('results/'+i)
        if not os.path.exists('model/'+i):
            os.makedirs('model/'+i)

    # (1) classical contro: with feature vectors
    if args.game in ['cartpole', 'mountaincar']:
        game = 'CartPole-v0'  if args.game == 'cartpole' else 'MountainCarMyEasyVersion-v0'
        if args.game == 'mountaincar':
            gym.envs.register(
                id='MountainCarMyEasyVersion-v0',
                entry_point='gym.envs.classic_control:MountainCarEnv',
                max_episode_steps=1000,
            )
        print('method {} on game {}!'.format(args.method, args.game))
        if args.method == 'DQN':
            if args.game == 'mountaincar':
                dqn_feature(game=game, n_step=1, replay_cls=UniformReplay, async_replay=True, noisy_linear=False)
            else:
                dqn_feature(game=game, n_step=1, replay_cls=UniformReplay, async_replay=True, noisy_linear=True)
        elif args.method == 'QRDQN': # change lr for two games
            quantile_regression_dqn_feature(game=game)
        else: # C51
            categorical_dqn_feature(game=game)


    # (2) Atari
    if args.game in ['breakout', 'qbert']:
        if args.game == 'breakout':
            game = 'BreakoutNoFrameskip-v4'
        else: # args.game == 'qbert':
            game = 'QbertNoFrameskip-v4'

        if args.method == 'DQN':
            dqn_pixel(game=game)
        elif args.method == 'QRDQN':
            quantile_regression_dqn_pixel(game=game)
        else: # C51
            categorical_dqn_pixel(game=game)
