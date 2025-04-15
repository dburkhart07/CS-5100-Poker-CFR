import random
import numpy as np
import clubs

class CFR_Agent:
    def __init__(self):
        self.regrets = {}
        self.strategy = {}
        self.opponent_tendencies = {}
        self.actions = ['fold', 'call', 'raise']
    
    def get_strategy(self, info_set):
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)
        
        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        self.strategy[info_set] = strategy
        
        return strategy
    
    def select_action(self, info_set):
        strategy = self.get_strategy(info_set)

        # Compute opponent fold rate (initialize all probabilities to even if information set is new)
        fold_rate = self.opponent_tendencies.get(info_set, [1, 1, 1])[0] / np.sum(self.opponent_tendencies.get(info_set, [1, 1, 1]))

        # More bluffing if opponent folds often (always 20% chance at least) - Adjustable
        bluff_probability = max(0.3, fold_rate / 2)

        # Randomly decide when to bluff (raise no matter what)
        if random.random() < bluff_probability:
            return 'raise'

        # Otherwise make calculated decision
        return np.random.choice(self.actions, p=strategy)

    
    def update_regrets(self, info_set, action_idx, regret, decay_rate=0.95):
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))
        self.regrets[info_set] *= decay_rate
        self.regrets[info_set][action_idx] = max(0, self.regrets[info_set][action_idx] + regret)    
    
    def update_opponent_tendencies(self, info_set, opponent_action_idx, decay_rate = 0.95):
        if info_set not in self.opponent_tendencies:
            self.opponent_tendencies[info_set] = np.zeros(len(self.actions))
        self.opponent_tendencies[info_set] *= decay_rate
        self.opponent_tendencies[info_set][opponent_action_idx] += 1


def train_cfr(agent, iterations=100000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    
    for _ in range(iterations):
        obs = dealer.reset(reset_stacks=True)
        history = []
        
        while True:
            # Extract relevant features
            hole_cards = str(obs['hole_cards'][0]) + str(obs['hole_cards'][1]) 
            community_cards = ''.join(str(card) for card in obs['community_cards'])
            pot_size = obs['pot']
            stacks = obs['stacks']
            street_commits = obs['street_commits']
            
            # Get the street based on the number of community cards
            num_community_cards = len(obs['community_cards'])
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            # Create the information set
            info_set = (hole_cards, community_cards, pot_size, street, tuple(stacks), tuple(street_commits))

            action = agent.select_action(info_set)
            action_idx = agent.actions.index(action)
            
            if action == 'fold':
                bet = 0
            elif action == 'call':
                bet = obs['call']
            else:
                # Consider making a bet based on the current hand strength as well (.evaluate function to get hand strength)
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                # Either go all in, raise minimum, or do half the pot
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))
            
            obs, rewards, done = dealer.step(bet)
            history.append((info_set, action_idx))
            
            if all(done):
                break
        
        # Update regrets based on the final outcome of the game
        for info_set, action_idx in history:
            for i, _ in enumerate(agent.actions):
                if i == action_idx:
                    regret = rewards[0]
                else:
                    # Negative for reward that was missed out
                    regret = -rewards[0]
                agent.update_regrets(info_set, i, regret)
                agent.update_opponent_tendencies(info_set, i)

def evaluate(agent, episodes=1000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)

    total_reward = 0
    
    for _ in range(episodes):
        obs = dealer.reset(reset_stacks=True)
        while True:
            # Extract relevant features
            hole_cards = str(obs['hole_cards'][0]) + str(obs['hole_cards'][1]) 
            community_cards = ''.join(str(card) for card in obs['community_cards'])
            pot_size = obs['pot']
            stacks = obs['stacks']
            street_commits = obs['street_commits']
            
            # Get the street based on the number of community cards
            num_community_cards = len(obs['community_cards'])
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            # Create the information set
            info_set = (hole_cards, community_cards, pot_size, street, tuple(stacks), tuple(street_commits))

            action = agent.select_action(info_set)
            if action == 'fold':
                bet = 0
            elif action == 'call':
                bet = obs['call']
            else:
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))
            obs, rewards, done = dealer.step(bet)
            if all(done):
                total_reward += rewards[0]
                break
    
    return total_reward / episodes


agent = CFR_Agent()
train_cfr(agent, iterations=30000)
avg_reward = evaluate(agent, episodes=10000)
# Yields average reward of around 3 chips (usually a bit higher) - substantially increases based on iterations
# 200 chips in total, wins on average 3 per round
print(f"Average reward after training CFR Regular agent: {avg_reward}")