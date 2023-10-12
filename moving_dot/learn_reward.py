import numpy as np
import random as random
import matplotlib.pyplot as plt
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from db import *
from log import *
from make_preferences_db import make_db


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RewardAgent(nn.Module):
    def __init__(self, input_size):
        super(RewardAgent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_reward(self, x):
        return self.network(x)
    
    def get_loss(self, segment1, segment2, mu1, mu2):
       
        reward_one = self.get_reward(segment1).squeeze(1)
        reward_two = self.get_reward(segment2).squeeze(1)

        e1 = torch.exp(reward_one)
        e2 = torch.exp(reward_two)

        P_pref_one = e1 / (e1 + e2)
        P_pref_two = e2 / (e1 + e2)

        loss = -(mu1 * torch.log(P_pref_one) + mu2 * torch.log(P_pref_two))

        return loss

def check_reward(agent, row):
    x1 = torch.tensor([row[1], row[2], row[3]], dtype=torch.float).to(device)
    x2 = torch.tensor([row[4], row[5], row[6]], dtype=torch.float).to(device)

    mu_1 = row[7]
    mu_2 = row[8]

    reward_one = agent.get_reward(x1)
    reward_two = agent.get_reward(x2)

    return reward_one.item(), reward_two.item(), mu_1, mu_2

def check_agent(agent, preference_table):

    in_sample = np.zeros([len(preference_table), 2])
    for i in range(len(preference_table)):
        row = preference_table[i]
        r = check_reward(agent, row)
        in_sample[i][0] = r[0] - r[1]
        in_sample[i][1] = r[2] - r[3]


    validation_conn = create_database_connection("validation_preferences.db")
    validation_preference_table = get_rows(validation_conn)

    validation = np.zeros([len(validation_preference_table), 2])
    for i in range(len(validation_preference_table)):
        row = validation_preference_table[i]
        r = check_reward(agent, row)
        validation[i][0] = r[0] - r[1]
        validation[i][1] = r[2] - r[3]

    neg = in_sample[in_sample[:,1] == -1, 0]
    mid = in_sample[in_sample[:,1] == 0, 0]
    pos = in_sample[in_sample[:,1] == 1, 0]

    neg_hist,neg_bins = np.histogram(neg)
    mid_hist, mid_bins = np.histogram(mid)
    pos_hist,pos_bins = np.histogram(pos)

    neg_hist = neg_hist / np.max(neg_hist)
    mid_hist = mid_hist / np.max(mid_hist)
    pos_hist = pos_hist / np.max(pos_hist)

    plt.hist(neg_bins[:-1], neg_bins, weights=neg_hist, alpha = 0.5, color="black", label = "s1 < s2")
    plt.hist(mid_bins[:-1], mid_bins, weights=mid_hist, alpha = 0.5, color = "magenta", label = "s1 = s2")
    plt.hist(pos_bins[:-1], pos_bins, weights=pos_hist, alpha = 0.5, color = "red", label = "s1 > s2")
    plt.legend()
    plt.show()

    validation_conn.close()
    return in_sample

if __name__ == "__main__":

    # first make the databases if they don't exist
    make_db("preferences.db")
    make_db("validation_preferences.db")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conn = create_database_connection()
    preference_table = get_rows(conn)

    agent = RewardAgent(2 + 1).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    checkpoint_intervals = [0, 1000, 10000, 25000, 100000]

    historical_loss = []
    in_sample_progress = []
    batch_size = 128
    for epoch in range(100000):

        entries = np.random.choice(len(preference_table), batch_size, False)
        batch = preference_table[entries]
        s1 = torch.tensor(batch[:,1:4], dtype = torch.float).to(device)
        s2 = torch.tensor(batch[:,4:7], dtype = torch.float).to(device)
        m1 = torch.tensor(batch[:,7], dtype = torch.float).to(device)
        m2 = torch.tensor(batch[:,8], dtype = torch.float).to(device)

        loss = agent.get_loss(s1, s2, m1, m2).sum()
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch % 100 == 0):
            log_blue(str(epoch) + " " + str(loss.item()))

        historical_loss.append(loss.item())

        if (epoch in checkpoint_intervals):
            in_sample = check_agent(agent, preference_table)
            in_sample_progress.append(in_sample)

    plt.plot(np.convolve(historical_loss, np.ones(100), 'valid') / 100)
    plt.show()
    torch.save(agent.state_dict(), "./rewards_model")

    check_agent(agent, preference_table)

