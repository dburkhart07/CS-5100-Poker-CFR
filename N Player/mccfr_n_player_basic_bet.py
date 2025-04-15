import numpy as np
import clubs
import random
from collections import defaultdict

class MCCFR_N_Player_Optimized_Bet:
    def __init__(self, decay_rate=0.95, bucket_size=500, use_bluffing=True):
        self.regrets = defaultdict(lambda: np.zeros(3))
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.strategy = defaultdict(lambda: np.ones(3)/3)
        self.actions = ['fold', 'call', 'raise']
        self.opponent_tendencies = defaultdict(lambda: np.zeros(3))
        self.hand_eval_cache = {}
        self.decay_rate = decay_rate
        self.bucket_size = bucket_size
        self.use_bluffing = use_bluffing
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)

    def get_strategy(self, info_set):
        regrets = self.regrets[info_set]
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        self.strategy_sum[info_set] += strategy
        self.strategy[info_set] = strategy
        return strategy

    def select_action(self, info_set, is_trained_player=True):
        # Generalized random possibilities (cater towards calling)
        if not is_trained_player:
            return np.random.choice(self.actions, p=[0.2, 0.5, 0.3])
            
        strategy = self.get_strategy(info_set)

        # Special bluffing
        if self.use_bluffing:
            total_actions = np.sum(self.opponent_tendencies[info_set])
            # Only consider bluffing after having 10 actions to go off of
            if total_actions > 10:
                # Bluff based on fold rate
                fold_rate = self.opponent_tendencies[info_set][0] / (total_actions + 1e-6)
                bluff_probability = min(0.3, fold_rate * 0.5)
                if np.random.random() < bluff_probability:
                    return 'raise'

        return np.random.choice(self.actions, p=strategy)

    def update_regrets(self, info_set, action_idx, regret):
        self.regrets[info_set][action_idx] += regret
        self.opponent_tendencies[info_set] *= self.decay_rate
        self.opponent_tendencies[info_set][action_idx] += 1

    def compute_average_strategy(self):
        # Average the entire strategy so far and assign that to be the new one
        for info_set in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[info_set])
            if normalizing_sum > 1e-8:
                self.strategy[info_set] = self.strategy_sum[info_set] / normalizing_sum
            else:
                self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)

    # Implore customized betting strategy
    def determine_bet_size(self, hand_strength, pot_size, min_raise, stack_size, street):      
        multipliers = {
            'flop': [0.6, 0.5, 0.4, 0.3],
            'turn': [0.7, 0.6, 0.5, 0.4],
            'river': [0.8, 0.7, 0.6, 0.5]
        }
        thresholds = [500, 2000, 3000]
        level = sum(hand_strength >= t for t in thresholds)
            
        if street == 'preflop':
            base_bet = max(min_raise, pot_size * 0.5)
        else:
            mult = multipliers.get(street, [0.5, 0.5, 0.5, 0.5])[min(level, 3)]  # Ensure level is within bounds
            base_bet = pot_size * mult
            
        # Ensure the bet is valid and within stack constraints
        bet = min(stack_size, max(min_raise, int(base_bet)))
        return bet
        
    def abstract_info_set(self, obs, hole, board, pot_size, street, street_commits, player_idx):
        stack_bucket = int(obs['stacks'][player_idx] / 50)  # Abstract stack into buckets
        pot_bucket = int(pot_size / 50) # Abstract pot size into buckets

        # Abstract hand strength into buckets (~ 32 levels per bucket)
        if street == 'preflop':
            hand_strength_bucket = 0
        else:
            raw_hand_strength = self.evaluator.evaluate(hole, list(board))
            hand_strength_bucket = int(raw_hand_strength / 250)

        return (hand_strength_bucket, pot_bucket, stack_bucket, street, street_commits)

def train_mccfr_n_player_basic_bet(agent, num_players=4, iterations=10000):
    blinds = [1, 2] + [0] * (num_players - 2)

    def reset_dealer():
        return clubs.poker.Dealer(
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

    dealer = reset_dealer()

    successful_iterations = 0

    while successful_iterations < iterations: 
        try:
            obs = dealer.reset(reset_stacks=True)
            histories = [[] for _ in range(num_players)]
            done = [False] * num_players
            
            while not all(done) and obs['action'] != -1:              
                current_player_idx = obs['action']
                hole = obs['hole_cards']
                board = obs['community_cards']
                pot = obs['pot']
                stacks = tuple(obs['stacks'])
                min_raise = obs['min_raise']
                street_commits = tuple(obs['street_commits'])

                street = ['preflop', 'flop', 'turn', 'river'][len(board) - 0 if len(board) == 0 else len(board) - 2]

                info_set = agent.abstract_info_set(obs, hole, board, pot, street, street_commits, current_player_idx)
                
                # For training, treat index 0 as the trained player
                is_trained_player = (current_player_idx == 0)
                action = agent.select_action(info_set, is_trained_player)
                action_idx = agent.actions.index(action)

                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = min(obs['call'], stacks[current_player_idx])
                else:
                    strength = agent.evaluator.evaluate(hole, board)
                    bet = agent.determine_bet_size(strength, pot, min_raise, stacks[current_player_idx], street)
                    
                histories[current_player_idx].append((info_set, action_idx))
                obs, rewards, done = dealer.step(bet)
            
            # Check if the hand actually completed
            if len(rewards) == num_players:
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
                
                successful_iterations += 1
            
            if successful_iterations % 1000 == 0:
                # Update every 1000 iterations
                agent.compute_average_strategy()
            
        except Exception:
            dealer = reset_dealer()
    
    # Compute final average at the end
    agent.compute_average_strategy()

def evaluate_vs_random(agent, num_players=4, episodes=1000, trained_player_idx=0):
    blinds = [1, 2] + [0] * (num_players - 2)

    def reset_dealer():
        return clubs.poker.Dealer(
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

    dealer = reset_dealer()
    total_rewards = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        try:
            obs = dealer.reset(reset_stacks=True)
            done = [False] * num_players

            while not all(done) and obs['action'] != -1:
                    
                current_player_idx = obs['action']
                
                if not obs['active'][current_player_idx]:
                    obs, _, done = dealer.step(-1)
                    continue

                hole = obs['hole_cards']
                board = obs['community_cards']
                pot = obs['pot']
                stacks = tuple(obs['stacks'])
                min_raise = obs['min_raise']
                street_commits = tuple(obs['street_commits'])

                street = ['preflop', 'flop', 'turn', 'river'][len(board) - 0 if len(board) == 0 else len(board) - 2]

                if current_player_idx == trained_player_idx:
                    info_set = agent.abstract_info_set(obs, hole, board, pot, street, street_commits, current_player_idx)
                    action = agent.select_action(info_set)
                else:
                    action = random.choice(agent.actions)

                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = min(obs['call'], stacks[current_player_idx])
                else:
                    strength = agent.evaluator.evaluate(hole, board)
                    bet = agent.determine_bet_size(strength, pot, min_raise, stacks[current_player_idx], street)

                obs, rewards, done = dealer.step(bet)

            if len(rewards) == num_players:
                total_rewards += rewards[0]
                completed_episodes += 1
                

        except Exception as e:
            dealer = reset_dealer()
            continue

    return total_rewards / completed_episodes


# NUM_PLAYERS = 4
# agent = MCCFR_N_Player_Optimized_Bet()

# print("Training MCCFR agent...")
# # Start with fewer iterations for testing
# train_mccfr_n_player_basic_bet(agent, num_players=NUM_PLAYERS, iterations=1000)

# print("Evaluating against random agents...")
# for _ in range(10):
#     avg_rewards = evaluate_vs_random(agent, num_players=NUM_PLAYERS, episodes=10000, trained_player_idx=0)

#     print(f"Average rewards vs random agents: {avg_rewards}")