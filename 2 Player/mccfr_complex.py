import numpy as np
import clubs

class MCCFR_Sampling_Agent:
    def __init__(self, epsilon=0.05, tau=1, beta=1e6):
        # Define regret and strategy variables
        self.regrets = {}
        self.strategy = {}
        self.strategy_sum = {}
        # Define exploration parameters
        self.epsilon = epsilon
        self.tau = tau
        self.beta = beta
        self.actions = ['fold', 'call', 'raise']
    
    def get_strategy(self, info_set):
        # Initialize the current and nash equilibrium strategy if need be
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)
            self.strategy_sum[info_set] = np.zeros(len(self.actions))
        
        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        # This is the sume of the positive regrets for each action to take in the given info set
        normalizing_sum = np.sum(positive_regrets)
        
        # Update strategy to be normalized version of the positive regrets
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        return strategy
    
    def sample_action(self, info_set):
        # Sum up the strategy to get total probability
        strategy = self.get_strategy(info_set)
        total = np.sum(strategy)
        # Use the paper's formula to calculate the sampling probabilities
        probabilities = [
            max(self.epsilon, self.beta + self.tau * strategy[a] / (self.beta + total)) for a in range(len(self.actions))
        ]
        # Normalize all 3 probabilities
        probabilities = np.array(probabilities) / np.sum(probabilities)
        # Select an action based on this sample
        return np.random.choice(self.actions, p=probabilities)
    
    def update_regrets(self, info_set, action_idx, regret):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))
        self.regrets[info_set][action_idx] += regret
    
    # Called at ver end to achieve nash equilibrium
    def compute_average_strategy(self):
        for info_set in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[info_set])
            if normalizing_sum > 0:
                self.strategy[info_set] = self.strategy_sum[info_set] / normalizing_sum


    # Used to determine bet size (only called after confirming we are betting)
    def determine_bet_size(self, hand_strength, pot_size, min_raise, stack_size, street):
        if street == 'flop':
            if hand_strength < 500:
                return min(stack_size, max(min_raise * 3, pot_size * 0.75))
            elif hand_strength < 2000:
                return min(stack_size, max(min_raise * 2.5, pot_size * 0.6))
            elif hand_strength < 3000:
                return min(stack_size, max(min_raise * 1.5, pot_size * 0.5))
            else:
                return min(stack_size, max(min_raise, pot_size * 0.4))
        
        elif street == 'turn':
            if hand_strength < 500:
                return min(stack_size, max(min_raise * 3, pot_size * 0.85))
            elif hand_strength < 2000:
                return min(stack_size, max(min_raise * 2.5, pot_size * 0.7))
            elif hand_strength < 3000:
                return min(stack_size, max(min_raise * 1.5, pot_size * 0.55))
            else:
                return min(stack_size, max(min_raise, pot_size * 0.45))
        
        elif street == 'river':
            if hand_strength < 500:
                return min(stack_size, max(min_raise * 3, pot_size * 0.9))
            elif hand_strength < 2000:
                return min(stack_size, max(min_raise * 2.5, pot_size * 0.75))
            elif hand_strength < 3000:
                return min(stack_size, max(min_raise * 1.5, pot_size * 0.5))
            else:
                return min(stack_size, max(min_raise, pot_size * 0.6))
        
        return min_raise


def train_mccfr_sampling(agent, iterations=100000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
    
    for _ in range(iterations):
        obs = dealer.reset(reset_stacks=True)
        history = []
        
        while True:
            hole_cards = obs['hole_cards']
            community_cards = obs['community_cards']
            hand_strength = evaluator.evaluate(hole_cards, community_cards)
            pot_size = obs['pot']
            stacks = obs['stacks']
            min_raise = obs['min_raise']
            street_commits = obs['street_commits']
            
            street = ['preflop', 'flop', 'turn', 'river'][len(community_cards) // 3]
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            
            action = agent.sample_action(info_set)
            action_idx = agent.actions.index(action)
            if action == 'fold':
                bet = 0
            elif action == 'call':
                bet = obs['call']
            else:
                bet = agent.determine_bet_size(hand_strength, pot_size, min_raise, stacks[0], street)
            
            obs, rewards, done = dealer.step(bet)
            history.append((info_set, action_idx, rewards[0]))
            
            if all(done):
                break
        
        for info_set, action_idx, reward in history:
            counterfactual_value = np.dot(agent.get_strategy(info_set), agent.regrets.get(info_set, np.zeros(len(agent.actions))))
            regret = reward - counterfactual_value
            agent.update_regrets(info_set, action_idx, regret)
    
    agent.compute_average_strategy()

def evaluate(agent, episodes=1000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
    total_reward = 0
    
    for _ in range(episodes):
        obs = dealer.reset(reset_stacks=True)
        
        while True:
            hole_cards = obs['hole_cards']
            community_cards = obs['community_cards']
            hand_strength = evaluator.evaluate(hole_cards, community_cards)
            pot_size = obs['pot']
            stacks = obs['stacks']
            min_raise = obs['min_raise']
            street_commits = obs['street_commits']
            
            street = ['preflop', 'flop', 'turn', 'river'][len(community_cards) // 3]
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            
            action = agent.sample_action(info_set)
            if action == 'fold':
                bet = 0
            elif action == 'call':
                bet = obs['call']
            else:
                bet = agent.determine_bet_size(hand_strength, pot_size, min_raise, stacks[0], street)
            
            obs, rewards, done = dealer.step(bet)
            if all(done):
                total_reward += rewards[0]
                break
    
    return total_reward / episodes

# # Train and evaluate the efficient MCCFR agent
# agent = MCCFR_Sampling_Agent()
# train_mccfr_sampling(agent, iterations=300000)
# avg_reward = evaluate(agent, episodes=100000)
# print(f"Average reward after training MCCFR Complex Sampling agent: {avg_reward}")
