import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import rl_playground.utils.plot_utils as plut

'''
The original implementation of REINFORCE from OpenAI's spinning-up blog.
This works well because it uses an extremely simple problem (cart-pole)
with continuous state but only 2 discrete actions.
'''

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

if __name__ == '__main__':
    env_name='CartPole-v0'
    hidden_sizes=[32]
    lr=1e-2 
    epochs=50
    batch_size=5000
    render=False

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name, render_mode="rgb_array")
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()[0]       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs)

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        W = np.array(batch_weights)
        ind_W = np.argsort(W)
        W_sorted = W[ind_W]
        X = torch.as_tensor(batch_obs, dtype=torch.float32)
        U = torch.as_tensor(batch_acts, dtype=torch.int32)
        Weights = torch.as_tensor(batch_weights, dtype=torch.float32)
        
        rn = np.arange(W.shape[0], dtype=int)
        pi = get_policy(X)
        prob_pre = pi.probs[rn, U].detach().numpy()
        logprob_pre = pi.log_prob(U).detach().numpy()
        weighted_logp_pre  = logprob_pre  * W
        ind_logp = np.argsort(weighted_logp_pre)
        weighted_logp_pre  = weighted_logp_pre[ind_logp]

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=X, act=U, weights=Weights)
        batch_loss.backward()
        optimizer.step()
        batch_loss_post = compute_loss(obs=X, act=U, weights=Weights)

        pi = get_policy(X)
        prob_post = pi.probs[rn, U].detach().numpy()
        logprob_post = pi.log_prob(U).detach().numpy()
        weighted_logp_post = logprob_post * W
        weighted_logp_post = weighted_logp_post[ind_logp]

        delta_logprob    = logprob_post - logprob_pre
        delta_prob       = prob_post - prob_pre
        # delta_logits  = logits_pre[ind_W] - logits_post[ind_W]

        # plt.figure()
        # plt.plot(W_sorted/np.max(W_sorted), 'o ', label="Normalized weights")
        # plt.plot(delta_prob[ind_W]/np.max(delta_prob), 'x ', label="Delta prob")
        # # plt.plot(delta_logits/np.max(delta_logits), 'x ', label="Delta logits")
        # plt.legend()

        # fig, ax = plut.create_empty_figure(3)
        # ax[0].plot(W[ind_logp], 'o ', label="Weights")
        # ax[0].legend()
        # ax[1].plot(logprob_pre[ind_logp], 'x ', label="Log Prob pre")
        # # ax[1].plot(logprob_post[ind_logp], 'o ', label="Log Prob post")
        # ax[1].plot(delta_logprob[ind_logp], 'o ', label="Delta Log Prob")
        # ax[1].legend()
        # ax[2].plot(prob_pre[ind_logp], 'x ', label="Prob pre")
        # # ax[2].plot(prob_post[ind_logp], 'o ', label="Prob post")
        # ax[2].plot(delta_prob[ind_logp], 'o ', label="Delta Prob")
        # ax[2].legend()

        # fig, ax = plut.create_empty_figure(2)
        # ax[0].plot(weighted_logp_pre,  label="weighted logp pre",  alpha=0.5)
        # ax[0].plot(weighted_logp_post, label="weighted logp post", alpha=0.5)
        # ax[0].plot([0, rn[-1]], 2*[np.mean(weighted_logp_pre)],  ':', label="Avg pre",  alpha=0.5)
        # ax[0].plot([0, rn[-1]], 2*[np.mean(weighted_logp_post)], ':', label="Avg post", alpha=0.5)
        # ax[0].legend()
        # ax[1].plot(weighted_logp_post-weighted_logp_pre, label="delta weighted logp")
        # ax[1].legend()
        # plt.show()
        

        return batch_loss, batch_loss_post, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, loss_post, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f delta loss: %.3f\t return: %.3f \t ep_len: %.3f'%
            (i, batch_loss, loss_post-batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
