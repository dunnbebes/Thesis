import os
import sys
            
# Create directories for models and logs
current_time           = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
models_dir             = f"models/keyref2{purpose}-{current_time}"
logdir                 = f"logs/keyref2{purpose}-{current_time}"
log_training_txt_dir   = "keyref2_log_training_txt"
log_training_excel_dir = "keyref2_log_training_excel"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(log_training_txt_dir):
    os.makedirs(log_training_txt_dir)
if not os.path.exists(log_training_excel_dir):
    os.makedirs(log_training_excel_dir)

# Generate unique file names based on current time
log_file          = os.path.join(log_training_txt_dir,   f"training_keyref2.txt")
excel_file        = os.path.join(log_training_excel_dir, f"training_keyref2.xlsx")
action_count_file = os.path.join(log_training_txt_dir,   f"action_count_keyref2.txt")
action_excel_file = os.path.join(log_training_excel_dir, f"action_count_keyref2.xlsx")


# Define the custom callback -------------------------------------------------------------
class CustomCallback(BaseCallback):
    def __init__(self, log_dir, excel_file, txt_file, action_count_file, action_excel_file, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.excel_file = excel_file
        self.txt_file = txt_file
        self.action_count_file = action_count_file
        self.action_excel_file = action_excel_file
        self.logs = []
        self.episode_rewards = []
        self.action_counts = {}
        self.episode_start = True

    def _on_training_start(self) -> None:
        # Initialize action counts
        self.action_counts = {action: 0 for action in action_list}

    def _on_step(self) -> bool:
        if self.episode_start:
            self.episode_rewards.append(0)
            self.episode_start = False

        # Record reward for the current step
        reward = self.locals['rewards'][0]
        self.episode_rewards[-1] += reward

        # Increment action count
        action = self.locals.get('actions', None)
        if action is not None:
            action_name = action_list[action[0]]
            self.action_counts[action_name] += 1
            
        return True

    def _on_rollout_end(self) -> None:
        # Called at the end of each episode
        sum_reward   = self.episode_rewards[-1] if self.episode_rewards else 0
        
        self.logger.record('train/episode_reward',   sum_reward)
        self.logs.append({
            'sum_reward': sum_reward,
        })

        self.episode_start = True
    
    def _on_training_end(self) -> None:
        # Save logs to Excel
        df = pd.DataFrame(self.logs)
        df.to_excel(self.excel_file, index=False)

        action_df = pd.DataFrame(list(self.action_counts.items()), columns=['Action', 'Count'])
        action_df.to_excel(self.action_excel_file, index=False)

        # Save logs to text file
        with open(self.txt_file, 'w') as f:
            f.write(df.to_string(index=False))
        with open(self.action_count_file, 'w') as f:
            f.write(action_df.to_string(index=False))


import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for DQN.
    
    :param observation_space: (spaces.Box)
    """
    def __init__(self, observation_space: spaces.Box):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim=6)  # Output features_dim matches last layer's output
        n_input_nodes = observation_space.shape[0]
        self.fc1 = nn.Linear(n_input_nodes, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        self.fc5 = nn.Linear(30, 30)
        self.fc6 = nn.Linear(30, 6)  # Output layer with 6 nodes

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.fc1(observations))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = self.fc6(x)  # Output layer
        return x

# Custom DQN class to implement soft target update
class CustomDQN(DQN):
    def __init__(self, *args, tau=0.01, **kwargs):
        super(CustomDQN, self).__init__(*args, **kwargs)
        self.tau = tau

    def train(self, gradient_steps, batch_size=100):
        # Train for gradient_steps
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Mix online and target networks
            target_q_values = self.q_net_target(replay_data.next_observations)
            next_q_values, _ = target_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)

            # Compute the target for the Q function
            target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations).gather(1, replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)

            # Optimize the model
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            # Soft update of target network
            with torch.no_grad():
                for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            self.logger.record('train/loss', loss.item())