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
        
        # Give a higher weight to choices that resulted in bigger rewards earlier
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        self.strategy[info_set] = strategy
        return strategy
    
    def select_action(self, info_set):
        strategy = self.get_strategy(info_set)
        fold_rate = self.opponent_tendencies.get(info_set, np.ones(len(self.actions)))[0] / np.sum(self.opponent_tendencies.get(info_set, np.ones(len(self.actions))))
        # Note: seems like raising it more and more always seems to work
        bluff_probability = max(0.3, fold_rate / 2)
        if random.random() < bluff_probability:
            return 'raise'
        return np.random.choice(self.actions, p=strategy)


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
            num_community_cards = len(obs['community_cards'])
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            action = agent.select_action(info_set)
            action_idx = agent.actions.index(action)
            
            if action == 'fold':
                bet = 0
            elif action == 'call':
                bet = obs['call']
            else:
                bet = agent.determine_bet_size(hand_strength, pot_size, min_raise, stacks[0], street)
            
            obs, rewards, done = dealer.step(bet)
            history.append((info_set, action_idx))
            
            if all(done):
                break
        
        for info_set, action_idx in history:
            for i in range(len(agent.actions)):
                # Update regret with how much we got and missed out on by the very end
                regret = rewards[0] if i == action_idx else -rewards[0]
                agent.update_regrets(info_set, i, regret)
                agent.update_opponent_tendencies(info_set, i)


def evaluate(agent, episodes=1000, render = False):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
    total_reward = 0
    
    for _ in range(episodes):
        obs = dealer.reset(reset_stacks=True)
        while True:
            if render:
                dealer.render(mode='ascii', sleep=0.5)

            hole_cards = obs['hole_cards']
            community_cards = obs['community_cards']
            hand_strength = evaluator.evaluate(hole_cards, community_cards)
            pot_size = obs['pot']
            stacks = obs['stacks']
            min_raise = obs['min_raise']
            street_commits = obs['street_commits']
            num_community_cards = len(obs['community_cards'])
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            action = agent.select_action(info_set)
            
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

# Converges to 2.4-2.5 chips/game
agent = CFR_Agent()
train_cfr(agent, iterations=100000)
avg_reward = evaluate(agent, episodes=100000)
print(f"Average reward after training CFR Betting agent: {avg_reward}")
