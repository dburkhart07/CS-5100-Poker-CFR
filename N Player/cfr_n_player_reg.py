import numpy as np
import clubs
import random

class CFRNPlayerAgent:
    def __init__(self):
        self.regrets = {}
        self.strategy = {}
        self.actions = ['fold', 'call', 'raise']
        # Decay rate for getting further into training
        self.decay_rate = 0.95
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)

    def get_strategy(self, info_set):
        # Initialize if not present
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)

        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        # Keep only positive regrets
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        # Regret-matching (update and normalize strategy based on regrets)
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        # Update strategy
        self.strategy[info_set] = strategy
        return strategy

    def select_action(self, info_set):
        # Choose action based on current strategy probabilities (default probabilities if not seen yet)
        strategy = self.get_strategy(info_set)
        return np.random.choice(self.actions, p=strategy)

    def update_regrets(self, info_set, action_idx, regret):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))

        self.regrets[info_set] *= self.decay_rate
        self.regrets[info_set][action_idx] += regret

    # Create an abstract information set
    def abstract_info_set(self, obs, player_idx):
        hole_cards = obs['hole_cards']
        board = obs['community_cards']
        street = ['preflop', 'flop', 'turn', 'river'][len(board) - 0 if len(board) == 0 else len(board) - 2]
        position = player_idx
        # Abstract the stack into buckets
        stack_bucket = int(obs['stacks'][player_idx] / 10) 
        # Abstract the pot size into buckets
        pot_bucket = int(obs['pot'] / 50)

        # Abstract the hand strength into buckets
        if street == 'preflop':
            hand_strength_bucket = 0
        else:
            raw_hand_strength = self.evaluator.evaluate(hole_cards, list(board))
            hand_strength_bucket = int(raw_hand_strength / 250)

        return (hand_strength_bucket, street, position, stack_bucket, pot_bucket)


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
                
                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = min(obs['call'], obs['stacks'][current_player])
                elif action == 'raise':
                    # Use a pot-size raise capped by player's stack
                    bet = min(obs['stacks'][current_player], max(obs['min_raise'], obs['pot'] // 2))

                # Keep track of the history for each player
                histories[current_player].append((info_set, action_idx))
                obs, rewards, done = dealer.step(bet)

            # Update regret after finishing a hand
            for player_idx in range(num_players):
                for info_set, taken_action_idx in histories[player_idx]:
                        # Current strategy
                        strategy = agent.get_strategy(info_set)

                        # Estimate counterfactual values: here we use a proxy — final reward
                        # Since we're doing outcome sampling, use the reward as a sampled estimate of v(I, a)
                        util = np.full(len(agent.actions), rewards[player_idx])

                        # Compute expected value across strategy
                        expected_util = np.dot(strategy, util)

                        # Update regrets for **all** actions
                        for a in range(len(agent.actions)):
                            regret = util[a] - expected_util
                            agent.update_regrets(info_set, a, regret)

        # Skip over episode as soon as anything goes wrong
        except Exception:
            continue


def evaluate_against_random(agent, num_players=4, episodes=1000):

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

    # Keep track ofthe reward for the agent (player 0)
    total_reward = 0

    for episode in range(episodes):
        try:
            obs = dealer.reset(reset_stacks=True)
            done = [False] * num_players

            while not all(done) and obs['action'] != -1:
                current_player = obs['action']

                # Strategy for CFR agent
                if current_player == 0:
                    info_set = agent.abstract_info_set(obs, current_player)
                    action = agent.select_action(info_set)

                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], obs['stacks'][current_player])
                    elif action == 'raise':
                        # Use a pot-size raise capped by player's stack
                        bet = min(obs['stacks'][current_player], max(obs['min_raise'], obs['pot'] // 2))
                # Strategy for random opponent
                else:
                    action = np.random.choice(['fold', 'call', 'raise'])
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], obs['stacks'][current_player])
                    elif action == 'raise':
                        # Use a pot-size raise capped by player's stack
                        bet = min(obs['stacks'][current_player], max(obs['min_raise'], obs['pot'] // 2))

                obs, rewards, done = dealer.step(bet)

            # Accumulate player 0 reward
            total_reward += rewards[0]

        except Exception:
            continue

    # Compute and return average reward
    return total_reward / episodes


# NUM_PLAYERS = 2
# agent = CFRNPlayerAgent()

# print("Training CFR agent")
# train_cfr(agent, num_players=NUM_PLAYERS, iterations=1000)

# print("Evaluating CFR agent")
# for _ in range(10):
#     avg_rewards = evaluate_against_random(agent, num_players=NUM_PLAYERS, episodes=3000)
#     print(f"Average rewards: {avg_rewards}")
