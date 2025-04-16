import random
import numpy as np
import clubs

# From the CFR 2 Player directory
from cfr_reg import CFR_Agent as Reg_CFR_Agent
from cfr_reg import train_cfr as train_reg_cfr

# From the MCCFR 2 Player directory
from mccfr_basic_bet import MCCFR_Agent as MCCFR_Bet_Agent
from mccfr_basic_bet import train_mccfr as train_bet_mccfr

from mccfr_basic_reg import MCCFR_Agent as MCCFR_Reg_Agent
from mccfr_basic_reg import train_mccfr as train_reg_mccfr

from mccfr_complex import MCCFR_Sampling_Agent as MCCFR_Complex
from mccfr_complex import train_mccfr_sampling as train_mccfr_sampling


class RandomAgent:
    def select_action(self, obs):
        return random.choice(['fold', 'call', 'raise'])
    
    def determine_bet_size(self, obs):
        return obs['min_raise']


class ConservativeAgent:
    def select_action(self, obs):
        actions = ['call', 'fold', 'raise']
        probabilities = [0.6, 0.2, 0.2]
        
        return random.choices(actions, probabilities)[0]
    
    def determine_bet_size(self, obs):
        return obs['pot'] * 0.5


class MatchingAgent:
    def select_action(self, obs):
        actions = ['call', 'fold', 'raise']
        probabilities = [0.2, 0.6, 0.2]
        
        return random.choices(actions, probabilities)[0]
    
    def determine_bet_size(self, obs):
        return obs['pot'] * 0.6


class AggressiveAgent:
    def select_action(self, obs):
        actions = ['call', 'fold', 'raise']
        probabilities = [0.2, 0.2, 0.6]
        
        return random.choices(actions, probabilities)[0]
    
    def determine_bet_size(self, obs):
        return obs['pot'] * 0.75


def play_head_to_head(agent1, agent2, episodes=1000):
    config = clubs.configs.NO_LIMIT_HOLDEM_TWO_PLAYER
    dealer = clubs.poker.Dealer(**config)
    evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
    total_reward_agent1 = 0 
    total_reward_agent2 = 0
    
    for _ in range(episodes):
        obs = dealer.reset(reset_stacks=True)
        players = [agent1, agent2]
        turn_tracker = 0
        
        while True:
            # Even turns played by CFR, odd by Other 
            agent = players[turn_tracker % 2]

            # How CFR agent plays
            if  isinstance(agent, Reg_CFR_Agent):
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
                    pot_size = obs['pot']
                    stack_size = obs['stacks'][0]
                    bet = min(stack_size, max(obs['min_raise'], pot_size // 2))
            
            # How MCCFR Basic agent plays
            elif isinstance(agent, MCCFR_Bet_Agent) or isinstance(agent, MCCFR_Reg_Agent):
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
                    if isinstance(agent, MCCFR_Bet_Agent):
                        bet = agent.determine_bet_size(hand_strength, pot_size, min_raise, stacks[0], street)
                    else:
                        pot_size = obs['pot']
                        stack_size = obs['stacks'][0]
                        bet = min(stack_size, max(obs['min_raise'], pot_size // 2))

            # How MCCFR Complex plays
            elif isinstance(agent, MCCFR_Complex):
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

            # How the other agents play
            else:
                action = agent.select_action(obs)
                bet = agent.determine_bet_size(obs) if action == 'raise' else (0 if action == 'fold' else obs['call'])

            obs, rewards, done = dealer.step(bet)
            # Increment turn by 1
            turn_tracker += 1
            if all(done):
                # Accumulate rewards for both agents
                total_reward_agent1 += rewards[0]
                total_reward_agent2 += rewards[1]
                break
    
    # Return average rewards for both agents
    return total_reward_agent1 / episodes, total_reward_agent2 / episodes



cfr_agent_reg = Reg_CFR_Agent()
train_reg_cfr(cfr_agent_reg, iterations=30000)

# Initialize MCCFR agents
mccfr_bet_agent = MCCFR_Bet_Agent()
train_bet_mccfr(mccfr_bet_agent, iterations=30000)

mccfr_reg_agent = MCCFR_Reg_Agent()
train_reg_mccfr(mccfr_reg_agent, iterations=30000)

mccfr_complex_agent = MCCFR_Complex()
train_mccfr_sampling(mccfr_complex_agent, iterations=30000)


def game(agent1, agent2, name1, name2, iterations=1000):
    agent1_vs_agent2 = play_head_to_head(agent1, agent2, iterations)
    print(f'{name1} Agent vs {name2} Agent - {name1} Reward: {agent1_vs_agent2[0]} {name2} Reward: {agent1_vs_agent2[1]}')


# Initialize the other agents
random_opponent = RandomAgent()
conservative_opponent = ConservativeAgent()
matcher_opponent = MatchingAgent()
aggressive_opponent = AggressiveAgent()


# CFR agents against the basic agents
game(cfr_agent_reg, random_opponent, "CFR Reg", "Random", 10000)
# game(cfr_agent_reg, conservative_opponent, "CFR Reg", "Conservative", 1000)
# game(cfr_agent_reg, matcher_opponent, "CFR Reg", "Calling", 1000)
# game(cfr_agent_reg, aggressive_opponent, "CFR Reg", "Aggressive", 1000)

# MCCFR Agents versus Random
game(mccfr_bet_agent, random_opponent, "MCCFR Bet", "Random", 10000)
game(mccfr_reg_agent, random_opponent, "MCCFR Reg", "Random", 10000)

# Basic MCCFR Agents versus CFR agents
# game(mccfr_bet_agent, cfr_agent_reg, "MCCFR Bet", "CFR Reg", 1000)
# game(mccfr_reg_agent, cfr_agent_reg, "MCCFR Reg", "CFR Reg", 1000)


# MCCFR Agents against themselves
# game(mccfr_bet_agent, mccfr_reg_agent, "MCCFR Bet", "MCCFR Reg", 1000)

# Complex MCCFR Agent versus Random
game(mccfr_complex_agent, random_opponent, "MCCFR Complex", "Random", 10000)

# Complex MCCFR Agent versus the basic MCCFR Agents
# game(mccfr_complex_agent, mccfr_bet_agent, "MCCFR Complex", "MCCFR Bet", 1000)
# game(mccfr_complex_agent, mccfr_reg_agent, "MCCFR Complex", "MCCFR Reg", 1000)

