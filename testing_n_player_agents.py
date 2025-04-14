import numpy as np
import clubs
from collections import defaultdict

# Import all agents and their training functions
from cfr_n_player_reg import CFRNPlayerAgent
from cfr_n_player_reg import train_cfr as train_cfr

from mccfr_n_player_basic_bet import MCCFR_N_Player_Optimized_Bet
from mccfr_n_player_basic_bet import train_mccfr_n_player_basic_bet

from mccfr_n_player_basic_reg import MCCFR_N_Player_Optimized_Reg
from mccfr_n_player_basic_reg import train_mccfr_n_player_basic_reg

from mccfr_n_player_complex import MCCFR_N_Player_Complex
from mccfr_n_player_complex import train_n_player_cfr as train_mccfr_n_player_complex

# Basic Agent definitions:
class RandomAgent:
    def __init__(self):
        self.actions = ['fold', 'call', 'raise']

    def select_action(self):
        return np.random.choice(self.actions)
    
    def determine_bet_size(self, pot_size, min_raise, stack_size, street=None):
        if street == 'preflop':
            return min(stack_size, max(min_raise * 2, pot_size // 2))
        elif street == 'flop':
            return min(stack_size, max(min_raise * 1.5, pot_size // 3))
        else:
            return min(stack_size, max(min_raise, pot_size // 2))
        

class AggressiveAgent:
    def __init__(self):
        self.actions = ['fold', 'call', 'raise']

    def select_action(self):
        probabilities = [0.1, 0.1, 0.8]
        
        return np.random.choice(self.actions, p=probabilities)
        

class NPlayerGameSimulator:
    def __init__(self, num_players=4, train_iters=1000):
        self.num_players = num_players
        self.blinds = [1, 2] + [0] * (num_players - 2)
        self.train_iters = train_iters

        # Initialize all agents
        self.agents = {
            'Aggressive_Agent': AggressiveAgent(),
            'CFR': CFRNPlayerAgent(),
            'MCCFR_Basic_Bet': MCCFR_N_Player_Optimized_Bet(),
            'MCCFR_Basic_Reg': MCCFR_N_Player_Optimized_Reg(),
            'MCCFR_Complex': MCCFR_N_Player_Complex(),
            'Random_Agent': RandomAgent(),
        }

        # Train all agents
        print("Training CFR agent...")
        train_cfr(self.agents['CFR'], num_players=num_players, iterations=train_iters)

        print("Training MCCFR Basic Bet agent...")
        train_mccfr_n_player_basic_bet(self.agents['MCCFR_Basic_Bet'], iterations=train_iters)

        print("Training MCCFR Basic Reg agent...")
        train_mccfr_n_player_basic_reg(self.agents['MCCFR_Basic_Reg'], iterations=train_iters)

        print("Training MCCFR Complex agent...")
        train_mccfr_n_player_complex(self.agents['MCCFR_Complex'], iterations=train_iters)

        self.init_dealer()
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)

    def init_dealer(self):
        self.dealer = clubs.poker.Dealer(
            num_players=self.num_players,
            num_streets=4,
            blinds=self.blinds,
            antes=0,
            raise_sizes='pot',
            num_raises=float('inf'),
            num_suits=4,
            num_ranks=13,
            num_hole_cards=2,
            mandatory_num_hole_cards=0,
            start_stack=500,
            num_community_cards=[0, 3, 1, 1],
            num_cards_for_hand=5
        )

    def play_episode(self):
        evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)
        obs = self.dealer.reset(reset_stacks=True)
        done = [False] * self.num_players

        agent_order = list(self.agents.keys())
        agent_assignments = {i: self.agents[agent_order[i]] for i in range(self.num_players)}

        actions_taken = [None] * self.num_players

        while not all(done) and obs['action'] not in [-1, None]:
            if obs['action'] is None:
                print("Warning: obs['action'] is None. Skipping episode.")
                return None

            current_player_idx = obs['action']

            if not obs['active'][current_player_idx]:
                obs, _, done = self.dealer.step(-1)  # Use -1 for inactive players
                continue

            agent = agent_assignments[current_player_idx]
            #stacks = tuple(obs['stacks'])
            current_stack = obs['stacks'][current_player_idx]
            hole = obs['hole_cards']
            board = obs['community_cards']
            
            pot_size = obs['pot']
            min_raise = obs['min_raise']
            street_commits = tuple(obs['street_commits'])

            street = (
                'preflop' if len(board) == 0 else
                'flop' if len(board) == 3 else
                'turn' if len(board) == 4 else 'river'
            )

            try:
                if isinstance(agent, MCCFR_N_Player_Complex):
                    for idx, action in enumerate(actions_taken):
                        if idx != current_player_idx and action is not None:
                            agent.update_opponent_profile(idx, action)

                    info_set = agent.abstract_info_set(obs, hole, board, pot_size, street, street_commits, current_player_idx)
                    
                    action = agent.sample_action(info_set)
                    
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        max_possible_raise = min(obs.get('max_raise', current_stack), current_stack)
                        if obs['min_raise'] > max_possible_raise:
                            bet = min(obs['call'], current_stack)
                        else:
                            bet = agent.determine_bet_size(pot_size, obs['min_raise'], current_stack)
                            bet = min(bet, max_possible_raise)

                elif isinstance(agent, MCCFR_N_Player_Optimized_Bet):
                    max_possible_raise = min(obs.get('max_raise', current_stack), current_stack)
                    
                    info_set = agent.abstract_info_set(obs, hole, board, pot_size, street, street_commits, current_player_idx)
                    
                    # Set position for bet sizing
                    agent.position = current_player_idx
                    action = agent.select_action(info_set, 1.0)

                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        # Pass the right parameters to determine_bet_size
                        strength = evaluator.evaluate(hole, board)
                        # Set the player's position for bet sizing
                        agent.position = current_player_idx
                        bet = bet = agent.determine_bet_size(strength, pot_size, min_raise, current_stack, street)

                elif isinstance(agent, MCCFR_N_Player_Optimized_Reg):
                    max_possible_raise = min(obs.get('max_raise', current_stack), current_stack)
                    
                    info_set = agent.abstract_info_set(obs, hole, board, pot_size, street, street_commits, current_player_idx)
                    
                    # Set position for bet sizing
                    agent.position = current_player_idx
                    action = agent.select_action(info_set, 1.0)

                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        bet = agent.determine_bet_size(pot_size, min_raise, current_stack)
                    

                elif isinstance(agent, CFRNPlayerAgent):
                    info_set = agent.abstract_info_set(obs, current_player_idx)  # Pass both obs and player_idx
                    action = agent.select_action(info_set)
                    bet = agent.get_bet_amount(obs, current_player_idx, action)  # Pass all required parameters


                elif isinstance(agent, RandomAgent) or isinstance(agent, AggressiveAgent):
                    action = agent.select_action()

                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        if isinstance(agent, RandomAgent):
                            max_possible_raise = min(obs['max_raise'], current_stack)
                            if obs['min_raise'] > max_possible_raise:
                                bet = min(obs['call'], current_stack)
                            else:
                                bet = agent.determine_bet_size(pot_size, obs['min_raise'], current_stack)
                                bet = min(bet, max_possible_raise)
                        else:
                            bet = pot_size * 0.75

                #print(f"current_player_idx: {current_player_idx}")
                action_idx = agent.actions.index(action)
                actions_taken[current_player_idx] = action_idx

                obs, rewards, done = self.dealer.step(bet)

            except Exception as e:
                # print(f"Error during step for player {current_player_idx}: {str(e)}")
                # # Print additional debugging info
                # print(f"Agent type: {type(agent).__name__}")
                
                # if isinstance(agent, MCCFR_N_Player_Optimized_Bet):
                #     print(f"hole type: {type(hole)}, board type: {type(board)}")
                #     if hasattr(hole, '__iter__'):
                #         print(f"hole is iterable, length: {len(hole)}")
                #     else:
                #         print("hole is not iterable")
                self.init_dealer()
                return None

        return rewards

    def simulate_games(self, episodes=10000):
        total_rewards = defaultdict(float)
        completed_episodes = 0
        agent_names = list(self.agents.keys())

        for episode in range(1, episodes + 1):
            rewards = self.play_episode()
            if rewards is None:
                continue  # Skip this episode
                
            completed_episodes += 1
            
            for i, name in enumerate(agent_names):
                if i < len(rewards):  # Make sure we have a reward for this agent
                    total_rewards[name] += rewards[i]
            
            if episode % 100 == 0:
                print(f"Completed {completed_episodes} valid episodes out of {episode} attempts")

        print("\nFinal Results:")
        avg_rewards = {}
        for name in agent_names:
            # Calculate average based on completed episodes
            if completed_episodes > 0:
                avg_rewards[name] = total_rewards[name] / completed_episodes
            else:
                avg_rewards[name] = 0.0
            print(f"{name}: {avg_rewards[name]:.2f} (Total: {total_rewards[name]:.2f})")

        return avg_rewards

n = 6
simulator = NPlayerGameSimulator(num_players=n)
print(f"\nStarting {n}-player simulation...")
avg_rewards = simulator.simulate_games(episodes=5000)
