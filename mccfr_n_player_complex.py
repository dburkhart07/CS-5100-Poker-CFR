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

        opp_tends = self.opponent_tendencies.get(info_set, [1,1,1])
        # Bluffing based on folding
        fold_rate = (opp_tends[0] + 1) / sum(opp_tends + 1)
        fold_bluff_prob = max(0.15, fold_rate/2)
        
        if np.random.random() < fold_bluff_prob:
            return 'raise'

        # Bluffing based on aggression factor
        aggression_factor = (opp_tends[2] + 1) / (opp_tends[1] + opp_tends[2] + 1)
        aggression_bluff_prob = max(0.15, aggression_factor / 2)
        if np.random.random() < aggression_bluff_prob:
            return 'raise'
    
        # In the case we choose not to bluff, performing action sampling
        total = np.sum(strategy)
        probabilities = [max(self.epsilon, self.beta + self.tau * strategy[a] / (self.beta + total)) for a in range(len(self.actions))]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        return np.random.choice(self.actions, p=probabilities)
    
    def update_regrets(self, info_set, action_idx, regret):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))

        # Discounted regret updates
        self.regrets[info_set] *= self.decay_rate  
        self.regrets[info_set][action_idx] += regret

        # Track opponent actions
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
        # Street-aware bet sizing
        if street == 'preflop':
            return min(stack_size, max(min_raise*2, pot_size//2))
        elif street == 'flop':
            return min(stack_size, max(min_raise*1.5, pot_size//3))
        else:  # turn/river
            return min(stack_size, max(min_raise, pot_size//2))
    
    

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
                stacks = obs['stacks']
                min_raise = obs['min_raise']
                street_commits = obs['street_commits']
                
                street = 'preflop' if len(community_cards) == 0 else \
                         'flop' if len(community_cards) == 3 else \
                         'turn' if len(community_cards) == 4 else 'river'
                
                info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
                
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


def evaluate(agent, num_players=4, episodes=1000):
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

    total_rewards = np.zeros(num_players)

    for _ in range(episodes):
        try:
            obs = dealer.reset(reset_stacks=True)
            done = [False] * num_players
            
            while not all(done) and obs['action'] != -1:
                current_player_idx = obs['action']
                
                if not obs['active'][current_player_idx]:
                    obs, _, done = dealer.step(0)
                    continue
                
                hole_cards = obs['hole_cards']
                community_cards = obs['community_cards']
                pot_size = obs['pot']
                stacks = obs['stacks']
                street_commits = obs['street_commits']
                
                street = 'preflop' if len(community_cards) == 0 else \
                         'flop' if len(community_cards) == 3 else \
                         'turn' if len(community_cards) == 4 else 'river'
                
                info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
                
                action = agent.sample_action(info_set)
                
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
            
            total_rewards += np.array(rewards)
        
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
    
# train_n_player_cfr(agent, num_players=NUM_PLAYERS, iterations=30000)
    
# print("Evaluating...")
# for _ in range(10):
#     avg_rewards = evaluate(agent, num_players=NUM_PLAYERS, episodes=10000)
#     print(f"Average rewards after training MCCFR N Player Complex Agent: {avg_rewards}")