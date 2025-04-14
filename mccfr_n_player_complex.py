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
        self.opponent_tendencies = {}
        self.decay_rate = decay_rate
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
        self.opponent_profiles = {} # Dict[player_idx] = [folds, calls, raises]

    def get_strategy(self, info_set):
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)
            self.strategy_sum[info_set] = np.zeros(len(self.actions))

        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)

        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        return strategy

    def sample_action(self, info_set):
        strategy = self.get_strategy(info_set)
        opp_tends = self.opponent_profiles.get(info_set, [1, 1, 1])

        fold_tendency = (opp_tends[0] + 1) / (sum(opp_tends) + 1)
        call_tendency = (opp_tends[1] + 1) / (sum(opp_tends) + 1)
        raise_tendency = (opp_tends[2] + 1) / (sum(opp_tends) + 1)

        fold_bluff_prob = max(0.15, fold_tendency / 2)
        if np.random.random() < fold_bluff_prob:
            return 'raise'

        aggression_factor = raise_tendency / (call_tendency + raise_tendency + 1)
        aggression_bluff_prob = max(0.15, aggression_factor / 2)
        if np.random.random() < aggression_bluff_prob:
            return 'raise'

        
        total = np.sum(strategy)
        probabilities = [
            max(self.epsilon, self.beta + self.tau * strategy[a] / (self.beta + total)) 
            for a in range(len(self.actions))
        ]
        
        probabilities[0] *= fold_tendency  # More likely to fold if the opponent folds more
        probabilities[2] *= aggression_factor  # More likely to raise if opponent is aggressive

        probabilities = np.array(probabilities) / np.sum(probabilities)
        
        return np.random.choice(self.actions, p=probabilities)


    def update_regrets(self, info_set, action_idx, regret):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))

        self.regrets[info_set] *= self.decay_rate
        self.regrets[info_set][action_idx] += regret

        if info_set not in self.opponent_tendencies:
            self.opponent_tendencies[info_set] = np.zeros(len(self.actions))
        self.opponent_tendencies[info_set] *= self.decay_rate
        self.opponent_tendencies[info_set][action_idx] += 1

    def compute_average_strategy(self):
        for info_set in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[info_set])
            if normalizing_sum > 1e-8:
                self.strategy[info_set] = self.strategy_sum[info_set] / normalizing_sum
            else:
                self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)

    def determine_bet_size(self, pot_size, min_raise, stack_size, street=None):
        if street == 'preflop':
            return min(stack_size, max(min_raise * 2, pot_size // 2))
        elif street == 'flop':
            return min(stack_size, max(min_raise * 1.5, pot_size // 3))
        else:
            return min(stack_size, max(min_raise, pot_size // 2))

    def abstract_info_set(self, obs, hole, community, pot_size, street, street_commits, player_idx):
        position = player_idx
        stack_bucket = int(obs['stacks'][player_idx] / 50)
        pot_bucket = int(pot_size / 50)

        if street == 'preflop':
            hand_strength_bucket = 0
        else:
            raw_hand_strength = self.evaluator.evaluate(hole, list(community))
            hand_strength_bucket = int(raw_hand_strength / 250)

        return (hand_strength_bucket, pot_bucket, stack_bucket, street, street_commits, position)
    
    def update_opponent_profile(self, player_idx, action_idx):
        if player_idx not in self.opponent_profiles:
            self.opponent_profiles[player_idx] = np.zeros(len(self.actions))
        
        self.opponent_profiles[player_idx][action_idx] += 1


    

# TODO: POTENTIALLY CAN MAKE SOME CONVERGENCE CHECKER (AFTER 20 FAILURES TO DO BETTER UPDATE)
def train_n_player_cfr(agent, num_players=4, iterations=100000):
    # Correct blinds configuration for 4 players: two blinds posted
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

    successful_iterations = 0
    
    while successful_iterations < iterations:
        # print(successful_iterations)
        try:
            obs = dealer.reset(reset_stacks=True)
            histories = [[] for _ in range(num_players)]
            done = [False] * num_players
            
            while not all(done) and obs['action'] != -1:
                current_player_idx = obs['action']
                
                if not obs['active'][current_player_idx]:
                    obs, rewards, done = dealer.step(0)
                    continue
                    
                hole_cards = obs['hole_cards']
                community_cards = obs['community_cards']
                pot_size = obs['pot']
                stacks = tuple(obs['stacks'])
                street_commits = tuple(obs['street_commits'])
                
                street = 'preflop' if len(community_cards) == 0 else \
                         'flop' if len(community_cards) == 3 else \
                         'turn' if len(community_cards) == 4 else 'river'

                info_set = agent.abstract_info_set(obs, hole_cards, community_cards, pot_size, street, street_commits, current_player_idx)
                
                action = agent.sample_action(info_set)
                action_idx = agent.actions.index(action)

                
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
                
                histories[current_player_idx].append((info_set, action_idx))
                
                # Log opponent behavior
                for idx in range(num_players):
                    if idx != current_player_idx:
                        agent.update_opponent_profile(idx, action_idx)

                try:
                    obs, rewards, done = dealer.step(bet)
                except Exception as e:
                    raise
            
            # Ensure rewards array has correct length before updating
            if len(rewards) != num_players:
                raise ValueError(f"Unexpected rewards length: {len(rewards)}")
            
            for player_idx in range(num_players):
                for info_set, action_idx in histories[player_idx]:
                    regret = rewards[player_idx]
                    agent.update_regrets(info_set, action_idx, regret)
            
            successful_iterations += 1
                
            if successful_iterations % 1000 == 0:
                agent.compute_average_strategy()
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
                
        except Exception as e:
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
            continue
    
    agent.compute_average_strategy()


def evaluate_against_random(agent, num_players=4, episodes=1000, trained_player_idx=0):
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

    total_rewards = 0

    for _ in range(episodes):
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
                    obs, _, done = dealer.step(0)
                    continue
                
                hole_cards = obs['hole_cards']
                board = obs['community_cards']
                pot_size = obs['pot']
                stacks = tuple(obs['stacks'])
                street_commits = tuple(obs['street_commits'])
                
                street = 'preflop' if len(board) == 0 else \
                         'flop' if len(board) == 3 else \
                         'turn' if len(board) == 4 else 'river'
                
                if current_player_idx == trained_player_idx:
                    info_set = agent.abstract_info_set(obs, hole_cards, board, pot_size, street, street_commits, current_player_idx)
                    action = agent.sample_action(info_set)
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
            continue

    return total_rewards / episodes


# NUM_PLAYERS = 4
# agent = MCCFR_N_Player_Complex()
    
# train_n_player_cfr(agent, num_players=NUM_PLAYERS, iterations=10000)
    
# print("Evaluating...")
# for _ in range(10):
#     avg_rewards = evaluate_against_random(agent, num_players=NUM_PLAYERS, episodes=10000, trained_player_idx=0)
#     print(f"Average rewards after training MCCFR N Player Complex Agent: {avg_rewards}")