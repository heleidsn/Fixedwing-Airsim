import sys
sys.path.append(r"C:\Users\helei\Documents\GitHub\PY_17_Sim\Fixedwing-Airsim\src")

from envs.obstacle_avoidance import ObstacleAvoidance
import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.utils.tensorboard import SummaryWriter
# import wandb

from utils.custom_policy_sb3 import CustomNoCNN, CustomCNN_GAP, CustomCNN_fc, CustomCNN_mobile
import torch as th

# wandb.init(project="SingleObserverTest")
# config = wandb.config
# config.dropout = 1.0

# writer = SummaryWriter()
env = ObstacleAvoidance()
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

feature_num_state = 2
feature_num_cnn = 25
policy_kwargs = dict(
    features_extractor_class=CustomNoCNN,
    features_extractor_kwargs=dict(features_dim=feature_num_state+feature_num_cnn),  # 指定最后total feature 数目 应该是CNN+state
    activation_fn=th.nn.ReLU
)
policy_kwargs['net_arch']=[64, 32]
model = TD3('CnnPolicy', env, 
            action_noise=action_noise,
            policy_kwargs=policy_kwargs, verbose=1,
            learning_starts=2000,
            batch_size=128,
            train_freq=(200, 'step'),
            gradient_steps=200,
            buffer_size=50000, seed=0,
            tensorboard_log="./td3_learning_tensorboard")

env.model = model
model.learn(total_timesteps=2e5)

# # wandb.watch(model)

obs = env.reset()
for _ in range(10):
    print("running")
    action = 10
    # action = env.action_space.sample()
    for t in range(10000):
        obs, rewards, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            obs = env.reset()
            break

env.render()
# writer.flush()
env.close()
