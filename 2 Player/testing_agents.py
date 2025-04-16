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


# Random agent 
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

            # Extract relevant features
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

            # Separate play for each type of agent

            if  isinstance(agent, Reg_CFR_Agent):  
                info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
                action = agent.select_action(info_set)
                
                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = obs['call']
                else:
                    pot_size = obs['pot']
                    stack_size = obs['stacks'][0]
                    bet = min(stack_size, max(obs['min_raise'], pot_size // 2))
            
            elif isinstance(agent, MCCFR_Bet_Agent) or isinstance(agent, MCCFR_Reg_Agent):  
                info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
                action = agent.select_action(info_set)
                
                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = obs['call']
                else:
                    # Different betting for MCCFR Bet vs MCCFR Reg
                    if isinstance(agent, MCCFR_Bet_Agent):
                        bet = agent.determine_bet_size(hand_strength, pot_size, min_raise, stacks[0], street)
                    else:
                        pot_size = obs['pot']
                        stack_size = obs['stacks'][0]
                        bet = min(stack_size, max(obs['min_raise'], pot_size // 2))

            elif isinstance(agent, MCCFR_Complex):
                info_set = (str(hole_cards), str(community_cards), pot_size, street, tuple(stacks), tuple(street_commits))
                
                action = agent.sample_action(info_set)
                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = obs['call']
                else:
                    bet = agent.determine_bet_size(hand_strength, pot_size, min_raise, stacks[0], street)

            # How the other agents play (random, aggressive, conservative, matching)
            else:
                action = agent.select_action(obs)
                if action == 'fold':
                    bet = -1
                elif action == 'call':
                    bet = obs['call']
                else:
                    bet = agent.determine_bet_size(obs)

            # Advance the game
            obs, rewards, done = dealer.step(bet)
            # Increment turn by 1 (other player's turn)
            turn_tracker += 1

            # Distribute rewards when the game ends
            if all(done):
                # Accumulate rewards for both agents
                total_reward_agent1 += rewards[0]
                total_reward_agent2 += rewards[1]
                break
    
    # Return average rewards for both agents
    return total_reward_agent1 / episodes, total_reward_agent2 / episodes



# Train necessary agents
train_iters = 30000

cfr_agent_reg = Reg_CFR_Agent()
train_reg_cfr(cfr_agent_reg, iterations=train_iters)

# Initialize MCCFR agents
mccfr_bet_agent = MCCFR_Bet_Agent()
train_bet_mccfr(mccfr_bet_agent, iterations=train_iters)

mccfr_reg_agent = MCCFR_Reg_Agent()
train_reg_mccfr(mccfr_reg_agent, iterations=train_iters)

mccfr_complex_agent = MCCFR_Complex()
train_mccfr_sampling(mccfr_complex_agent, iterations=train_iters)


# Play a head-to-head game between two agents
def game(agent1, agent2, name1, name2, iterations=1000):
    agent1_vs_agent2 = play_head_to_head(agent1, agent2, iterations)
    print(f'{name1} Agent vs {name2} Agent - {name1} Reward: {agent1_vs_agent2[0]} {name2} Reward: {agent1_vs_agent2[1]}')


# Initialize the other agents
random_opponent = RandomAgent()
conservative_opponent = ConservativeAgent()
matcher_opponent = MatchingAgent()
aggressive_opponent = AggressiveAgent()


game_episodes = 1000
# CFR agents against the basic agents
game(cfr_agent_reg, random_opponent, "CFR Reg", "Random", game_episodes)
game(cfr_agent_reg, conservative_opponent, "CFR Reg", "Conservative", game_episodes)
game(cfr_agent_reg, matcher_opponent, "CFR Reg", "Calling", game_episodes)
game(cfr_agent_reg, aggressive_opponent, "CFR Reg", "Aggressive", game_episodes)

# MCCFR Agents versus Random
game(mccfr_bet_agent, random_opponent, "MCCFR Bet", "Random", game_episodes)
game(mccfr_reg_agent, random_opponent, "MCCFR Reg", "Random", game_episodes)

# Basic MCCFR Agents versus CFR agents
game(mccfr_bet_agent, cfr_agent_reg, "MCCFR Bet", "CFR Reg", game_episodes)
game(mccfr_reg_agent, cfr_agent_reg, "MCCFR Reg", "CFR Reg", game_episodes)


# MCCFR Agents against themselves
game(mccfr_bet_agent, mccfr_reg_agent, "MCCFR Bet", "MCCFR Reg", game_episodes)

# Complex MCCFR Agent versus Random
game(mccfr_complex_agent, random_opponent, "MCCFR Complex", "Random", game_episodes)

# Complex MCCFR Agent versus the basic MCCFR Agents
game(mccfr_complex_agent, mccfr_bet_agent, "MCCFR Complex", "MCCFR Bet", game_episodes)
game(mccfr_complex_agent, mccfr_reg_agent, "MCCFR Complex", "MCCFR Reg", game_episodes)

