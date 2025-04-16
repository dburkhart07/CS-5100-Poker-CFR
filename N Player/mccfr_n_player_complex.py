import numpy as np
import clubs

class MCCFR_N_Player_Complex:
    def __init__(self, epsilon=0.05, tau=1, beta=1e6, decay_rate=0.95):
        self.regrets = {}
        self.strategy = {}
        self.strategy_sum = {}
        self.epsilon = epsilon
        self.tau = tau
        self.beta = beta
        self.actions = ['fold', 'call', 'raise']
        self.decay_rate = decay_rate
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
        # Dictionary for each opponent index of their folds, calls, and raises
        self.opponent_profiles = {}

    def get_strategy(self, info_set):
        # Initialize new strategy and total strategy if new state
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)
            self.strategy_sum[info_set] = np.zeros(len(self.actions))

        # Get and normalize all positive regrets (regret matching)
        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        # Add new strategy to accumulated one and assign as new strategy
        self.strategy_sum[info_set] += strategy
        self.strategy[info_set] = strategy

        return strategy

    def sample_action(self, info_set):
        strategy = self.get_strategy(info_set)
        opp_tends = self.opponent_profiles.get(info_set, [1, 1, 1])

        # Get the tendency for each action
        fold_tendency = (opp_tends[0] + 1) / (sum(opp_tends) + 1)
        call_tendency = (opp_tends[1] + 1) / (sum(opp_tends) + 1)
        raise_tendency = (opp_tends[2] + 1) / (sum(opp_tends) + 1)

        # Small probability to bluff someone who folds too much
        fold_bluff_prob = max(0.15, fold_tendency / 2)
        if np.random.random() < fold_bluff_prob:
            return 'raise'
        
        # Probability to bluff someone who raises too much
        aggression_factor = raise_tendency / (call_tendency + raise_tendency + 1)
        aggression_bluff_prob = max(0.15, aggression_factor / 2)
        if np.random.random() < aggression_bluff_prob:
            return 'raise'

        # Average strategy sampling
        total = np.sum(strategy)
        probabilities = [
            max(self.epsilon, self.beta + self.tau * strategy[a] / (self.beta + total)) 
            for a in range(len(self.actions))
        ]
        
        # Adjust probabilities for each tendency
        probabilities[0] *= fold_tendency
        probabilities[2] *= aggression_factor

        # Normalize
        probabilities = np.array(probabilities) / np.sum(probabilities)
        
        return np.random.choice(self.actions, p=probabilities)


    def update_regrets(self, info_set, action_idx, regret):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))

        self.regrets[info_set] *= self.decay_rate
        self.regrets[info_set][action_idx] += regret

    def compute_average_strategy(self):
        # Average the entire strategy so far and assign that to be the new one
        for info_set in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[info_set])
            if normalizing_sum > 1e-8:
                self.strategy[info_set] = self.strategy_sum[info_set] / normalizing_sum
            else:
                self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)

    # Relatively basic betting function
    def determine_bet_size(self, pot_size, min_raise, stack_size, street=None):
        if street == 'preflop':
            return min(stack_size, max(min_raise * 2, pot_size // 2))
        elif street == 'flop':
            return min(stack_size, max(min_raise * 1.5, pot_size // 3))
        else:
            return min(stack_size, max(min_raise, pot_size // 2))

    # Create an abstract information set
    def abstract_info_set(self, obs, hole, board, pot_size, street, street_commits, player_idx):
        # 8 buckets for stacks, pot size, and hand strength
        stack_bucket = min(int(obs['stacks'][player_idx] / 25), 8)
        pot_bucket = min(int(pot_size / 25), 8)

        if street == 'preflop':
            hand_strength_bucket = 0
        else:
            norm_strength = 7500 - self.evaluator.evaluate(hole, list(board))
            hand_strength_bucket = int(norm_strength / 1000) 

        return (hand_strength_bucket, pot_bucket, stack_bucket, street, street_commits)
    
    # Update the opponent profiles based on the action they took
    def update_opponent_profile(self, player_idx, action_idx):
        if player_idx not in self.opponent_profiles:
            self.opponent_profiles[player_idx] = np.zeros(len(self.actions))
        
        self.opponent_profiles[player_idx][action_idx] += 1


def train_n_player_cfr(agent, num_players=4, iterations=100000):
    # Initialize dealer and blinds
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
    
    # Only count successful iterations
    while successful_iterations < iterations:
        try:
            # Reset everything
            obs = dealer.reset(reset_stacks=True)
            histories = [[] for _ in range(num_players)]
            done = [False] * num_players
            
            # Play until all players are done or invalid action is entered (unlikely)
            while not all(done) and obs['action'] != -1:
                current_player_idx = obs['action']
                # Extract relevant features to create abstract state 
                hole_cards = obs['hole_cards']
                board = obs['community_cards']
                pot_size = obs['pot']
                stacks = tuple(obs['stacks'])
                street_commits = tuple(obs['street_commits'])
                
                street = ['preflop', 'flop', 'turn', 'river'][len(board) - 0 if len(board) == 0 else len(board) - 2]

                info_set = agent.abstract_info_set(obs, hole_cards, board, pot_size, street, street_commits, current_player_idx)
                
                # Sample action and get its index
                action = agent.sample_action(info_set)
                action_idx = agent.actions.index(action)

                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = min(obs['call'], stacks[current_player_idx])
                else:
                    # Either go all in, or use betting function
                    max_possible_raise = min(obs['max_raise'], stacks[current_player_idx])
                    if obs['min_raise'] > max_possible_raise:
                        bet = min(obs['call'], stacks[current_player_idx])
                    else:
                        bet = agent.determine_bet_size(pot_size, obs['min_raise'], stacks[current_player_idx])
                        bet = min(bet, max_possible_raise)
                
                histories[current_player_idx].append((info_set, action_idx))
                
                # Update opponent behavior
                for idx in range(num_players):
                    if idx != current_player_idx:
                        agent.update_opponent_profile(idx, action_idx)

                # Advance game
                try:
                    obs, rewards, done = dealer.step(bet)
                except Exception as e:
                    raise
            
            # Outcome sampling to measure regrets
            for player_idx in range(num_players):
                for info_set, taken_action_idx in histories[player_idx]:
                        # Current strategy
                        strategy = agent.get_strategy(info_set)

                        # Estimate counterfactual values: here we use a proxy — final reward
                        # Since we're doing outcome sampling, use the reward as a sampled estimate of v(I, a)
                        util = np.full(len(agent.actions), rewards[player_idx])

                        # Compute expected value across strategy
                        expected_util = np.dot(strategy, util)

                        # Update regrets for the taken action
                        regret = util[taken_action_idx] - expected_util
                        agent.update_regrets(info_set, taken_action_idx, regret)
            
            successful_iterations += 1
                
            if successful_iterations % 1000 == 0:
                # Reset the dealer and compute average strategy every 1000 iterations
                agent.compute_average_strategy()
                dealer = reset_dealer()
                
        except Exception as e:
            dealer =  reset_dealer()
            continue

    # Compute the final average strategy at the end
    agent.compute_average_strategy()


def evaluate_against_random(agent, num_players=4, episodes=1000, trained_player_idx=0):
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

    # Testing code is very similar to traniing code
    for _ in range(episodes):
        try:
            obs = dealer.reset(reset_stacks=True)
            done = [False] * num_players
            
            while not all(done) and obs['action'] != -1:
                current_player_idx = obs['action']

                if not obs['active'][current_player_idx]:
                    obs, _, done = dealer.step(-1)
                    continue
                
                hole_cards = obs['hole_cards']
                board = obs['community_cards']
                pot_size = obs['pot']
                stacks = tuple(obs['stacks'])
                street_commits = tuple(obs['street_commits'])
                
                street = ['preflop', 'flop', 'turn', 'river'][len(board) - 0 if len(board) == 0 else len(board) - 2]
                
                # For the trained agent, perform similar training steps to get the action
                if current_player_idx == trained_player_idx:
                    info_set = agent.abstract_info_set(obs, hole_cards, board, pot_size, street, street_commits, current_player_idx)
                    action = agent.sample_action(info_set)
                # On opponent moves, choose a random action and update the profile
                else:
                    action = np.random.choice(agent.actions)
                    action_idx = agent.actions.index(action)
                    agent.update_opponent_profile(current_player_idx, action_idx)
                
                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = min(obs['call'], stacks[current_player_idx])
                else:
                    max_possible_raise = min(obs['max_raise'], stacks[current_player_idx])
                    if obs['min_raise'] > max_possible_raise:
                        bet = min(obs['call'], stacks[current_player_idx])
                    else:
                        bet = agent.determine_bet_size(pot_size, obs['min_raise'], stacks[current_player_idx])
                        bet = min(bet, max_possible_raise)
                
                obs, rewards, done = dealer.step(bet)
            
            total_rewards += rewards[0]
        
        except Exception as e:
            dealer = reset_dealer()
            continue

    return total_rewards / episodes


# ===========================================================================================
# Training for the agent in just this file - UNCOMMENT TO EXPERIMENT WITH THIS AGENT'S POLICY
# ===========================================================================================
# NUM_PLAYERS = 4
# agent = MCCFR_N_Player_Complex()

# print("Training MCCFR agent")
# train_n_player_cfr(agent, num_players=NUM_PLAYERS, iterations=1000)
    
# print("Evaluating MCCFR agent")
# for _ in range(10):
#     avg_rewards = evaluate_against_random(agent, num_players=NUM_PLAYERS, episodes=10000, trained_player_idx=0)
#     print(f"Average rewards: {avg_rewards}")