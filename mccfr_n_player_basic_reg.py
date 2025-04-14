import numpy as np
import clubs
import random
from collections import defaultdict

class MCCFR_N_Player_Optimized_Reg:
    def __init__(self, decay_rate=0.95, bucket_size=500, use_bluffing=True, position=None):
        self.regrets = defaultdict(lambda: np.zeros(3))
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.strategy = defaultdict(lambda: np.ones(3)/3)
        self.actions = ['fold', 'call', 'raise']
        self.opponent_tendencies = defaultdict(lambda: np.zeros(3))
        self.hand_eval_cache = {}
        self.decay_rate = decay_rate
        self.bucket_size = bucket_size
        self.use_bluffing = use_bluffing
        self.position = position if position is not None else 0  # Default position to 0 if None
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)


    def get_strategy(self, info_set, reach_prob):
        regrets = self.regrets[info_set]
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        self.strategy_sum[info_set] += reach_prob * strategy
        self.strategy[info_set] = strategy
        return strategy

    def select_action(self, info_set, reach_prob, is_trained_player=True):
        if not is_trained_player:
            # Simplified strategy for opponents during training
            return np.random.choice(self.actions, p=[0.2, 0.5, 0.3])
            
        strategy = self.get_strategy(info_set, reach_prob)

        if self.use_bluffing:
            total_actions = np.sum(self.opponent_tendencies[info_set])
            if total_actions > 10:
                fold_rate = self.opponent_tendencies[info_set][0] / (total_actions + 1e-6)
                bluff_probability = min(0.3, fold_rate * 0.5)
                if random.random() < bluff_probability:
                    return 'raise'

        return np.random.choice(self.actions, p=strategy)

    def update_regrets(self, info_set, action_idx, regret):
        clipped_regret = np.clip(regret, -1000, 1000)
        self.regrets[info_set][action_idx] += clipped_regret
        self.opponent_tendencies[info_set] *= self.decay_rate
        self.opponent_tendencies[info_set][action_idx] += 1

    def compute_strategy(self):
        for info_set in self.strategy_sum:
            norm = np.sum(self.strategy_sum[info_set])
            if norm > 0:
                self.strategy[info_set] = self.strategy_sum[info_set] / norm

    def get_hand_bucket(self, hole_cards, community_cards, evaluator):
        try:
            def card_to_tuple(card):
                return (card.rank, card.suit)
            
            hole_tuples = tuple(sorted(card_to_tuple(card) for card in hole_cards))
            community_tuples = tuple(sorted(card_to_tuple(card) for card in community_cards))
            key = (hole_tuples, community_tuples)

            if key not in self.hand_eval_cache:
                strength = evaluator.evaluate(hole_cards, community_cards)
                self.hand_eval_cache[key] = strength

            return self.hand_eval_cache[key] // self.bucket_size
        except Exception as e:
            print(f"Error in get_hand_bucket: {str(e)}")
            # Return a default bucket if there's an error
            return 0

    def determine_bet_size(self, pot_size, min_raise, stack):
        # Compute desired raise
        desired_raise = max(min_raise, pot_size // 2)
        raise_amount = min(stack, desired_raise)

        # Return -1 if raise is not possible
        if raise_amount < min_raise:
            return -1  # fallback to fold
        
        return raise_amount

    def abstract_info_set(self, obs, hole, community, pot_size, street, street_commits, player_idx):
        position = player_idx
        stack_bucket = int(obs['stacks'][player_idx] / 50)  # Abstract stack into buckets
        pot_bucket = int(pot_size / 50) # Abstract pot size into buckets

        # Abstract hand strength into buckets (~ 32 levels per bucket)
        if street == 'preflop':
            hand_strength_bucket = 0
        else:
            raw_hand_strength = self.evaluator.evaluate(hole, list(community))
            hand_strength_bucket = int(raw_hand_strength / 250)

        return (hand_strength_bucket, pot_bucket, stack_bucket, street, street_commits, position)

def train_mccfr_n_player_basic_reg(agent, num_players=4, iterations=100000, verbose=True):
    blinds = [1, 2] + [0] * (num_players - 2)

    def reset_dealer():
        try:
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
        except Exception as e:
            print(f"Error creating dealer: {str(e)}")
            raise

    try:
        dealer = reset_dealer()
    except Exception as e:
        print(f"Failed to initialize dealer or evaluator: {str(e)}")
        return

    successful_iterations = 0
    max_retries = 5  # Maximum retries for each iteration if an error occurs

    while successful_iterations < iterations:
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            
            try:
                obs = dealer.reset(reset_stacks=True)
                histories = [[] for _ in range(num_players)]
                done = [False] * num_players
                step_count = 0
                
                while not all(done) and obs['action'] != -1:
                    step_count += 1
                    if step_count > 100:  # Safeguard against infinite loops
                        break
                        
                    current_player_idx = obs['action']
                    
                    if not obs['active'][current_player_idx]:
                        obs, rewards, done = dealer.step(-1)
                        continue

                    hole = obs['hole_cards']
                    board = obs['community_cards']
                    pot = obs['pot']
                    stacks = tuple(obs['stacks'])
                    min_raise = obs['min_raise']
                    street_commits = tuple(obs['street_commits'])

                    street = 'preflop' if len(board) == 0 else \
                             'flop' if len(board) == 3 else \
                             'turn' if len(board) == 4 else 'river'

                    info_set = agent.abstract_info_set(obs, hole, board, pot, street, street_commits, current_player_idx)
                    
                    # For training, treat index 0 as the trained player
                    is_trained_player = (current_player_idx == 0)
                    action = agent.select_action(info_set, 1.0, is_trained_player)
                    action_idx = agent.actions.index(action)

                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], stacks[current_player_idx])
                    else:
                        # Set the player's position for bet sizing
                        agent.position = current_player_idx
                        bet = agent.determine_bet_size(pot, min_raise, stacks[current_player_idx])
                        
                    histories[current_player_idx].append((info_set, action_idx))
                    obs, rewards, done = dealer.step(bet)
                
                # Check if the hand actually completed
                if len(rewards) == num_players:
                    for player_idx in range(num_players):
                        for info_set, action_idx in histories[player_idx]:
                            v_s = rewards[player_idx]
                            strategy = agent.strategy.get(info_set, np.ones(3)/3)
                            v_s_prime = np.dot(strategy, agent.regrets[info_set])
                            sampling_prob = strategy[action_idx] + 1e-6
                            regret = (v_s - v_s_prime) / sampling_prob
                            agent.update_regrets(info_set, action_idx, regret)
                    
                    successful_iterations += 1
                    success = True
                else:
                    if verbose:
                        print("Hand did not complete properly, retrying...")
                    retries += 1
                
                if successful_iterations % 1000 == 0:
                    agent.compute_strategy()
                
            except Exception as e:
                retries += 1
                #print(f"Error in iteration: {str(e)}")
                dealer = reset_dealer()
        
        if not success:
            print(f"Failed to complete iteration after {max_retries} retries, continuing...")
    
    agent.compute_strategy()
    return agent

def evaluate_vs_random(agent, num_players=4, episodes=1000, trained_player_idx=0, verbose=False):
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
            step_count = 0

            while not all(done) and obs['action'] != -1:
                step_count += 1
                if step_count > 100:  # Safeguard against infinite loops
                    break
                    
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

                street = 'preflop' if len(board) == 0 else \
                         'flop' if len(board) == 3 else \
                         'turn' if len(board) == 4 else 'river'

                if current_player_idx == trained_player_idx:
                    info_set = agent.abstract_info_set(obs, hole, board, pot, street, street_commits, current_player_idx)
                    # Set position for bet sizing
                    agent.position = current_player_idx
                    action = agent.select_action(info_set, 1.0)
                else:
                    action = random.choice(agent.actions)

                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = min(obs['call'], stacks[current_player_idx])
                else:
                    # Set the player's position for bet sizing
                    agent.position = current_player_idx
                    bet = agent.determine_bet_size(pot, min_raise, stacks[current_player_idx])

                obs, rewards, done = dealer.step(bet)

            if step_count <= 100 and len(rewards) == num_players:
                total_rewards += rewards[0]
                completed_episodes += 1
                

        except Exception as e:
            dealer = reset_dealer()
            continue

    return total_rewards / completed_episodes

# NUM_PLAYERS = 4
# # Create agent with a position specified
# agent = MCCFR_N_Player_Optimized_Reg(position=0)

# print("Training MCCFR agent...")
# # Start with fewer iterations for testing
# train_mccfr_n_player_basic_reg(agent, num_players=NUM_PLAYERS, iterations=10000, verbose=True)

# print("Evaluating against random agents...")
# for _ in range(10):
#     avg_rewards = evaluate_vs_random(agent, num_players=NUM_PLAYERS, episodes=10000, trained_player_idx=0, verbose=True)

#     print(f"Average rewards vs random agents: {avg_rewards}")