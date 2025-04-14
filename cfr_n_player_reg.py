import numpy as np
import random
import clubs

class CFRNPlayerAgent:
    def __init__(self):
        self.regrets = {}
        self.strategy = {}
        self.opponent_tendencies = {}
        self.actions = ['fold', 'call', 'raise']
        self.decay_rate = 0.98  # Slightly slower decay to retain useful history
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)

    def get_strategy(self, info_set):
        # Initialize if not present
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)

        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        # Regret-matching strategy computation
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        self.strategy[info_set] = strategy
        return strategy

    def select_action(self, info_set):
        strategy = self.get_strategy(info_set)
        return np.random.choice(self.actions, p=strategy)

    def update_regrets(self, info_set, action_taken, actual_utility, all_utilities):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))

        # Apply regret matching based on utility differences
        self.regrets[info_set] *= self.decay_rate
        for i in range(len(self.actions)):
            regret = all_utilities[i] - actual_utility
            self.regrets[info_set][i] += regret

    def update_opponent_tendencies(self, info_set, action_idx):
        if info_set not in self.opponent_tendencies:
            self.opponent_tendencies[info_set] = np.zeros(len(self.actions))

        self.opponent_tendencies[info_set] *= self.decay_rate
        self.opponent_tendencies[info_set][action_idx] += 1

    def abstract_info_set(self, obs, player_idx):
        hole_cards = obs['hole_cards']
        community = obs['community_cards']
        street = ['preflop', 'flop', 'turn', 'river'][len(community) - 0 if len(community) == 0 else len(community) - 2]
        position = player_idx
        stack_bucket = int(obs['stacks'][player_idx] / 10)  # Abstract stack into buckets
        pot_bucket = int(obs['pot'] / 50) # Abstract pot size into buckets

        if street == 'preflop':
            hand_strength_bucket = 0
        else:
            raw_hand_strength = self.evaluator.evaluate(hole_cards, list(community))
            hand_strength_bucket = int(raw_hand_strength / 250)

        return (hand_strength_bucket, street, position, stack_bucket, pot_bucket)

    def get_bet_amount(self, obs, player_idx, action):
        if action == 'fold':
            return -1
        elif action == 'call':
            return min(obs['call'], obs['stacks'][player_idx])
        elif action == 'raise':
            # Use a pot-size raise capped by player's stack
            return min(obs['stacks'][player_idx], max(obs['min_raise'], obs['pot'] // 2))
        return 0


def train_cfr(agent, num_players=4, iterations=100000):
    blinds = [1, 2] + [0] * (num_players - 2)
    dealer = clubs.poker.Dealer(
        num_players=num_players,
        num_streets=4,
        blinds=blinds,
        antes=0,
        raise_sizes='pot',
        num_raises=float('inf'),
        num_suits=4,
        num_ranks=13,
        num_hole_cards=2,
        mandatory_num_hole_cards=0,
        start_stack=200,
        num_community_cards=[0, 3, 1, 1],
        num_cards_for_hand=5
    )

    for _ in range(iterations):
        try:
            obs = dealer.reset(reset_stacks=True)
            histories = [[] for _ in range(num_players)]
            done = [False] * num_players

            while not all(done) and obs['action'] != -1:
                current_player = obs['action']

                if not obs['active'][current_player]:
                    obs, _, done = dealer.step(0)
                    continue

                info_set = agent.abstract_info_set(obs, current_player)
                action = agent.select_action(info_set)
                action_idx = agent.actions.index(action)
                bet = agent.get_bet_amount(obs, current_player, action)

                histories[current_player].append((info_set, action_idx))
                obs, rewards, done = dealer.step(bet)

            # Regret update after hand resolution
            for player_idx in range(num_players):
                final_reward = rewards[player_idx]
                for info_set, action_idx in histories[player_idx]:
                    utilities = [-final_reward if i != action_idx else final_reward for i in range(3)]
                    agent.update_regrets(info_set, action_idx, final_reward, utilities)
                    agent.update_opponent_tendencies(info_set, action_idx)

        except Exception:
            continue


def evaluate_against_random(agent, num_players=4, episodes=1000):
    from random import choice

    blinds = [1, 2] + [0] * (num_players - 2)
    dealer = clubs.poker.Dealer(
        num_players=num_players,
        num_streets=4,
        blinds=blinds,
        antes=0,
        raise_sizes='pot',
        num_raises=float('inf'),
        num_suits=4,
        num_ranks=13,
        num_hole_cards=2,
        mandatory_num_hole_cards=0,
        start_stack=200,
        num_community_cards=[0, 3, 1, 1],
        num_cards_for_hand=5
    )

    total_reward = 0

    for episode in range(episodes):
        try:
            obs = dealer.reset(reset_stacks=True)
            done = [False] * num_players

            while not all(done) and obs['action'] != -1:
                current_player = obs['action']

                if not obs['active'][current_player]:
                    obs, _, done = dealer.step(0)
                    continue

                if current_player == 0:
                    # CFR Agent plays as player 0
                    info_set = agent.abstract_info_set(obs, current_player)
                    action = agent.select_action(info_set)
                    bet = agent.get_bet_amount(obs, current_player, action)
                else:
                    # Random opponent
                    action = choice(['fold', 'call', 'raise'])
                    bet = agent.get_bet_amount(obs, current_player, action)

                obs, rewards, done = dealer.step(bet)

            total_reward += rewards[0]  # Only track CFR agent's reward

        except Exception:
            continue

    return total_reward / episodes


# NUM_PLAYERS = 4
# agent = CFRNPlayerAgent()

# print("Training CFR agent...")
# train_cfr(agent, num_players=NUM_PLAYERS, iterations=1000)

# print("Evaluating CFR agent...")
# for _ in range(10):
#     avg_rewards = evaluate_against_random(agent, num_players=NUM_PLAYERS, episodes=3000)
#     print(f"Average rewards: {avg_rewards}")
