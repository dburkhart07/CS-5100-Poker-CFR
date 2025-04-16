import numpy as np
import clubs

class MCCFR_Agent:
    def __init__(self):
        self.regrets = {}
        self.strategy = {}
        self.strategy_sum = {}
        self.actions = ['fold', 'call', 'raise']
    
    # Get the strategy for the agent based on the given information set
    def get_strategy(self, info_set):
        # Create even probability strategy if not otherwise there
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)
        
        # Get the regrets at the given information set (only keep positive ones)
        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        # Create the new strategy based on the normalized positive regrets
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        self.strategy[info_set] = strategy
        
        return strategy
    
    def select_action(self, info_set):
        strategy = self.get_strategy(info_set)
        # Randomly choose an action based on I (randomly choosing from the 3 a in Sigma(a|I)
        return np.random.choice(self.actions, p=strategy)
    
    def update_regrets(self, info_set, action_idx, regret):
        # Create new regret for each action if not already created
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))
        # Zero out regrets if it goes negative (avoid negative regrets)
        self.regrets[info_set][action_idx] += max(0, self.regrets[info_set][action_idx] + regret)  
    
    def compute_strategy(self):
        # For each info set, apply strategy based on accumulated regret
        for info_set in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[info_set])
            if normalizing_sum > 0:
                self.strategy[info_set] = self.strategy_sum[info_set] / normalizing_sum
    

def train_mccfr(agent, iterations=100000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    
    for _ in range(iterations):
        obs = dealer.reset(reset_stacks=True)
        history = []
        
        while True:
            # Extract relevant features
            hole_cards = obs['hole_cards']
            community_cards = obs['community_cards']
            pot_size = obs['pot']
            stacks = obs['stacks']
            street_commits = obs['street_commits']
            
            # Get the street based on the number of community cards
            num_community_cards = len(community_cards)
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            # Create state
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            
            # Use the state to get the action and action index
            action = agent.select_action(info_set)
            action_idx = agent.actions.index(action)
            
            if action == 'fold':
                bet = -1
            elif action == 'call':
                bet = obs['call']
            else:
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))

            # Advance game
            obs, rewards, done = dealer.step(bet)
            history.append((info_set, action_idx, rewards[0]))
            
            if all(done):
                break
        
        for info_set, action_idx, reward in history:
            # Terminal state value - v(s)
            v_s = reward
            # E(X) = Sum(p(x) * x), where p(x) is the probability of taking actions, and x is all possibile regrets at the state I
            v_s_prime_a = sum(agent.strategy[info_set] * np.array(agent.regrets.get(info_set, np.zeros(len(agent.actions)))))
            # Regret is how we did versus how we expected to do
            regret = v_s - v_s_prime_a
            # Update that regret we have for taking that action at I
            agent.update_regrets(info_set, action_idx, regret)
    
    # At the very end, normalize all the strategies
    agent.compute_strategy()

def evaluate(agent, episodes=1000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    total_reward = 0
    
    for _ in range(episodes):
        obs = dealer.reset(reset_stacks=True)
        while True:
            # Extract relevant features
            hole_cards = obs['hole_cards']
            community_cards = obs['community_cards']
            pot_size = obs['pot']
            stacks = obs['stacks']
            street_commits = obs['street_commits']
            
            # Determine street based on number of community cards
            num_community_cards = len(community_cards)
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            # Create informatino set
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            # Get action based on information set
            action = agent.select_action(info_set)
            
            if action == 'fold':
                bet = -1
            elif action == 'call':
                bet = obs['call']
            else:
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))

            # Advance game
            obs, rewards, done = dealer.step(bet)
            # Distribute rewards if game is over
            if all(done):
                total_reward += rewards[0]
                break
    
    # Return average reward
    return total_reward / episodes


# ===========================================================================================
# Training for the agent in just this file - UNCOMMENT TO EXPERIMENT WITH THIS AGENT'S POLICY
# ===========================================================================================
# agent = MCCFR_Agent()
# train_mccfr(agent, iterations=300000)
# avg_reward = evaluate(agent, episodes=100000)
# print(f"Average reward after training MCCFR Regular Agent: {avg_reward}")
