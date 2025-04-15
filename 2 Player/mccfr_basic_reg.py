import numpy as np
import clubs

class MCCFR_Agent:
    def __init__(self):
        self.regrets = {}
        self.strategy = {}
        self.strategy_sum = {}
        self.actions = ['fold', 'call', 'raise']
    
    def get_strategy(self, info_set, reach_prob):
        # Initialize a uniform strategy (probability) if new information set
        if info_set not in self.strategy:
            self.strategy[info_set] = np.ones(len(self.actions)) / len(self.actions)
            self.strategy_sum[info_set] = np.zeros(len(self.actions))
        
        # Regret dictates how often we choose a strategy - R(I,a)
        regrets = self.regrets.get(info_set, np.zeros(len(self.actions)))
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.ones(len(self.actions)) / len(self.actions)

        # This strategy is Sigma(a|I)
        self.strategy[info_set] = strategy
        # Reach prob is the prob of reaching info set I (used to get the average strategy at the end to normalize it)
        self.strategy_sum[info_set] += reach_prob * strategy
        return strategy
    
    def select_action(self, info_set, reach_prob):
        strategy = self.get_strategy(info_set, reach_prob)
        # Randomly choose an action based on I (randomly choosing from the 3 a in Sigma(a|I)
        return np.random.choice(self.actions, p=strategy)
    
    def update_regrets(self, info_set, action_idx, regret):
        # Regret is r(I,a)
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(len(self.actions))
        # Adding to R_t+1 (I,a) - constantly being updated to refine strategy
        self.regrets[info_set][action_idx] += regret
    
    def compute_strategy(self):
        # For each info set, apply strategy based on accumulated regret
        # Note: Regret dictates the strategy updates
        for info_set in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[info_set])
            if normalizing_sum > 0:
                self.strategy[info_set] = self.strategy_sum[info_set] / normalizing_sum    

def train_mccfr(agent, iterations=100000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
    
    for _ in range(iterations):
        obs = dealer.reset(reset_stacks=True)
        history = []
        reach_prob = 1.0
        
        while True:
            hole_cards = obs['hole_cards']
            community_cards = obs['community_cards']
            hand_strength = evaluator.evaluate(hole_cards, community_cards)
            pot_size = obs['pot']
            stacks = obs['stacks']
            min_raise = obs['min_raise']
            street_commits = obs['street_commits']
            
            num_community_cards = len(community_cards)
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            action = agent.select_action(info_set, reach_prob)
            action_idx = agent.actions.index(action)
            
            if action == 'fold':
                bet = 0
            elif action == 'call':
                bet = obs['call']
            else:
                pot_size = obs['pot']
                stack_size = obs['stacks'][0]
                bet = min(stack_size, max(obs['min_raise'], pot_size // 2))

            obs, rewards, done = dealer.step(bet)
            history.append((info_set, action_idx, rewards[0]))
            
            if all(done):
                break
        
        for info_set, action_idx, reward in history:
            # Terminal state value - v(s)
            v_s = reward
            # POTENTIALLY NOT BEING CALCULATED CORRECTLY??

            # E(X) = Sum(p(x) * x), where p(x) is the probability of taking actions, and x is all possibile regrets at the state I
            v_s_prime_a = sum(agent.strategy[info_set] * np.array(agent.regrets.get(info_set, np.zeros(len(agent.actions)))))
            # Regret is how we did versus how we expected to do
            regret = v_s - v_s_prime_a
            # Update that regret we have for taking that action at I
            agent.update_regrets(info_set, action_idx, regret)
    
    # At the very end, normalize all the strategies
    agent.compute_strategy()

def evaluate(agent, episodes=1000):
    """
    Step 6: Test the trained strategy.
    """
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
            
            num_community_cards = len(community_cards)
            if num_community_cards == 0:
                street = 'preflop'
            elif num_community_cards == 3:
                street = 'flop'
            elif num_community_cards == 4:
                street = 'turn'
            else:
                street = 'river'
            
            info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
            action = agent.select_action(info_set, 1.0)
            
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

# # Train and evaluate MCCFR agent
# # Averages around 5.5-6.5 chips/game
# agent = MCCFR_Agent()
# train_mccfr(agent, iterations=300000)
# avg_reward = evaluate(agent, episodes=100000)
# print(f"Average reward after training MCCFR Regular agent: {avg_reward}")
