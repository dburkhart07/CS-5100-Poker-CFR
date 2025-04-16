import random
import numpy as np
import clubs

class CFR_Agent:
    # Initialize important game attributes for the agent
    def __init__(self):
        self.regrets = {}
        self.strategy = {}
        self.opponent_tendencies = {}
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
    
    # Select an action based on the strategy
    def select_action(self, info_set):
        strategy = self.get_strategy(info_set)

        # Compute opponent fold rate based on how often the opponent folds
        fold_rate = self.opponent_tendencies.get(info_set, [1, 1, 1])[0] / np.sum(self.opponent_tendencies.get(info_set, [1, 1, 1]))

        # Random check to for bluffing
        bluff_probability = max(0.3, fold_rate / 2)
        if random.random() < bluff_probability:
            return 'raise'

        # Otherwise make calculated decision
        return np.random.choice(self.actions, p=strategy)

    
    def update_regrets(self, info_set, action_idx, regret, decay_rate=0.95):
        # Create new regret for each action if not already created
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))
        # Decay the regret as game goes further on (not as valuable later)
        self.regrets[info_set] *= decay_rate
        # Zero out regrets if it goes negative (avoid negative regrets)
        self.regrets[info_set][action_idx] = max(0, self.regrets[info_set][action_idx] + regret)    
    
    def update_opponent_tendencies(self, info_set, opponent_action_idx, decay_rate = 0.95):
        # Create new opponent tendency for each action if not already created
        if info_set not in self.opponent_tendencies:
            self.opponent_tendencies[info_set] = np.zeros(len(self.actions))
        # Decay the tendencies as the game goes on (not as valuable later)
        self.opponent_tendencies[info_set] *= decay_rate
        self.opponent_tendencies[info_set][opponent_action_idx] += 1


def train_cfr(agent, iterations=100000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    
    for _ in range(iterations):
        # Reset the game for all players
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

            # Get the action to take and index
            action = agent.select_action(info_set)
            action_idx = agent.actions.index(action)
            
            if action == 'fold':
                bet = -1
            elif action == 'call':
                bet = obs['call']
            else:
                # Either go all in, raise minimum, or do half the pot - only go all in if thats all thats left
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))
            
            # Advance the game
            obs, rewards, done = dealer.step(bet)
            # Add the state and action to the history
            history.append((info_set, action_idx))
            
            if all(done):
                break
        
        # Update regrets based on the final outcome of the game (outcome sampling)
        for info_set, action_idx in history:
            for i, _ in enumerate(agent.actions):
                # Reward gained
                if i == action_idx:
                    regret = rewards[0]
                # Reward lost
                else:
                    regret = -rewards[0]
                # Update each regret accordingly
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

            # Get the action to take
            action = agent.select_action(info_set)

            if action == 'fold':
                bet = -1
            elif action == 'call':
                bet = obs['call']
            else:
                # Either go all in, raise minimum, or do half the pot - only go all in if thats all thats left
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))
            
            # Advance the game
            obs, rewards, done = dealer.step(bet)
            # Distribute rewards if the game is over
            if all(done):
                total_reward += rewards[0]
                break
    
    # Compute the average reward
    return total_reward / episodes


# ===========================================================================================
# Training for the agent in just this file - UNCOMMENT TO EXPERIMENT WITH THIS AGENT'S POLICY
# ===========================================================================================
# agent = CFR_Agent()
# train_cfr(agent, iterations=3000)
# avg_reward = evaluate(agent, episodes=10000)
# print(f"Average reward after training CFR Regular agent: {avg_reward}")