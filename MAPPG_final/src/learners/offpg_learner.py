import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop


class OffPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = OffPGCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, log):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        #build q
        inputs = self.critic._build_inputs(batch, bs, max_t)
        q_vals = self.critic.forward(inputs).detach()[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs, _ = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        #compute parameters sum for debugging
        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(log[key])/ts_logged, t_env)
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", self.mac.agent.comm_fact, t_env)
            self.logger.log_stat("agent_parameter", p_sum, t_env)
            self.log_stats_t = t_env

    def train_critic(self, on_batch, best_batch=None, log=None):
        bs = on_batch.batch_size
        max_t = on_batch.max_seq_length
        rewards = on_batch["reward"][:, :-1]
        actions = on_batch["actions"][:, :]
        terminated = on_batch["terminated"][:, :-1].float()
        mask = on_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])



        #build_target_q
        target_inputs = self.target_critic._build_inputs(on_batch, bs, max_t)
        target_q_vals = self.target_critic.forward(target_inputs).detach()
        targets_taken = th.mean(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), dim=2, keepdim=True)
        target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda).repeat(1, 1, self.n_agents)

        inputs = self.critic._build_inputs(on_batch, bs, max_t)

        if best_batch is not None:
            best_target_q, best_inputs, best_mask, best_actions= self.train_critic_best(best_batch)
            target_q = th.cat((target_q, best_target_q), dim=0)
            inputs = th.cat((inputs, best_inputs), dim=0)
            mask = th.cat((mask, best_mask), dim=0)
            actions = th.cat((actions, best_actions), dim=0)

        mask = mask.repeat(1, 1, self.n_agents)

        #train critic
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals = self.critic.forward(inputs[:, t:t+1])
            q_vals = th.gather(q_vals, 3, index=actions[:, t:t+1]).squeeze(3)
            target_q_t = target_q[:, t:t+1]
            q_err = (q_vals - target_q_t) * mask_t
            critic_loss = (q_err ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            log["critic_loss"].append(critic_loss.item())
            log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            log["td_error_abs"].append((q_err.abs().sum().item() / mask_elems))
            log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            log["q_taken_mean"].append((q_vals * mask_t).sum().item() / mask_elems)

        #update target network
        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps



    def train_critic_best(self, batch):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:]

        # pr for all actions of the episode
        mac_out = []
        self.mac.init_hidden(bs)
        for i in range(max_t):
            agent_outs, _ = self.mac.forward(batch, t=i)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1).detach()
        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0
        critic_mac = th.gather(mac_out, 3, actions).squeeze(3).prod(dim=2, keepdim=True)

        #target_q take
        target_inputs = self.target_critic._build_inputs(batch, bs, max_t)
        target_q_vals = self.target_critic.forward(target_inputs).detach()
        targets_taken = th.mean(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), dim=2, keepdim=True)

        #expected q
        exp_q = self.build_exp_q(batch, mac_out, bs, max_t).detach()
        # td-error
        targets_taken[:, -1] = targets_taken[:, -1] * (1 - th.sum(terminated, dim=1))
        exp_q[:, -1] = exp_q[:, -1] * (1 - th.sum(terminated, dim=1))
        targets_taken[:, :-1] = targets_taken[:, :-1] * mask
        exp_q[:, :-1] = exp_q[:, :-1] * mask
        td_q = (rewards + self.args.gamma * exp_q[:, 1:] - targets_taken[:, :-1]) * mask

        #compute target
        target_q =  build_target_q(td_q, targets_taken[:, :-1], critic_mac, mask, self.args.gamma, self.args.td_lambda, self.args.step).detach().repeat(1, 1, self.n_agents)

        inputs = self.critic._build_inputs(batch, bs, max_t)

        return target_q, inputs, mask, actions






    def build_exp_q(self, batch, mac_out, bs, max_t):

        # inputs for target net
        inputs = []
        # state, obs, action
        inputs.append(batch["state"][:].unsqueeze(2).unsqueeze(2).repeat(1, 1, self.args.n_sum, self.n_agents, 1))
        inputs.append(batch["obs"][:].unsqueeze(2).repeat(1, 1, self.args.n_sum, 1, 1))
        # Sample n_sum number of possible actions and use importance sampling
        ac_sampler = Categorical(mac_out.unsqueeze(2).repeat(1, 1, self.args.n_sum, 1, 1) + 1e-10)
        actions = ac_sampler.sample().long().unsqueeze(4)
        action_one_hot = mac_out.new_zeros(bs, max_t, self.args.n_sum, self.n_agents, self.n_actions)
        action_one_hot = action_one_hot.scatter_(-1, actions, 1.0).view(bs, max_t, self.args.n_sum, 1, -1).repeat(1, 1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(action_one_hot * (agent_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)))
        # obs last action
        l_actions = batch["actions_onehot"][:].view(bs, max_t, 1, -1)
        if self.args.obs_last_action:
            last_action = []
            last_action.append(l_actions[:, 0:1])
            last_action.append(l_actions[:, :-1])
            last_action = th.cat([x for x in last_action], dim = 1)
            inputs.append(last_action.unsqueeze(2).repeat(1, 1, self.args.n_sum, self.n_agents, 1))

        #agent id
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, max_t, self.args.n_sum, -1, -1))
        inputs = th.cat([x for x in inputs], dim=-1)
        #E(V(s))
        target_exp_q_vals = self.target_critic.forward(inputs).detach()
        target_exp_q_vals = th.gather(target_exp_q_vals, 4, actions).squeeze(-1).mean(dim=3)
        action_mac = mac_out.unsqueeze(2).repeat(1, 1, self.args.n_sum, 1, 1)
        action_mac = th.gather(action_mac, 4, actions).squeeze(-1)
        action_mac = th.prod(action_mac, 3)
        target_exp_q_vals = th.sum(target_exp_q_vals * action_mac, dim=2, keepdim=True) / (th.sum(action_mac, dim=2, keepdim=True) + 1e-10)
        # target_exp_q_vals = th.sum(target_exp_q_vals, dim=2, keepdim=True) / self.args.n_sum
        return target_exp_q_vals


    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
