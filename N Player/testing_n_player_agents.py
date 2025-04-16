import numpy as np
import matplotlib.pyplot as plt
import clubs
import time
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

    # Take a completely random action
    def select_action(self):
        return np.random.choice(self.actions)
    
    # Basic betting function based on state of the game
    def determine_bet_size(self, pot_size, min_raise, stack_size):
        return min(stack_size, max(min_raise, pot_size // 2))


class AggressiveAgent:
    def __init__(self):
        self.actions = ['fold', 'call', 'raise']

    # Agent that favors towards betting 80% of the time
    def select_action(self):
        probabilities = [0.1, 0.1, 0.8]
        return np.random.choice(self.actions, p=probabilities)



class NPlayerGameSimulator:
    def __init__(self, num_players=4, train_iters=1000):
        self.num_players = num_players

        # First player is small blind, second is big blind
        self.blinds = [1, 2] + [0] * (num_players - 2)
        self.train_iters = train_iters

        # Keep track of average reward periodically for visualization purposes
        self.reward_history = defaultdict(list)

        # NOTE: ADJUST NUMBER OF PLAYERS ACCORDINGLY, ALSO TRAIN AGENTS THAT NEED TRAINING
        self.agents = {
            'CFR': CFRNPlayerAgent(),
            'MCCFR_Basic_Bet': MCCFR_N_Player_Optimized_Bet(),
            #'Aggressive_Agent': AggressiveAgent(),
            'MCCFR_Basic_Reg': MCCFR_N_Player_Optimized_Reg(),
            'MCCFR_Complex': MCCFR_N_Player_Complex(),
            #'#Random_Agent': RandomAgent(),
        }

        # Train all agents that have training functions
        print("Training CFR agent...")
        train_cfr(self.agents['CFR'], num_players=num_players, iterations=train_iters)

        print("Training MCCFR Basic Bet agent...")
        train_mccfr_n_player_basic_bet(self.agents['MCCFR_Basic_Bet'], iterations=train_iters)

        print("Training MCCFR Basic Reg agent...")
        train_mccfr_n_player_basic_reg(self.agents['MCCFR_Basic_Reg'], iterations=train_iters)

        print("Training MCCFR Complex agent...")
        train_mccfr_n_player_complex(self.agents['MCCFR_Complex'], iterations=train_iters)

        # Initialize the dealer and evaluator for hand strength
        self.init_dealer()
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)

    def init_dealer(self):
        # Relatively default setup for traditional No-Limit Texas Hold'Em
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
            start_stack=200,
            num_community_cards=[0, 3, 1, 1],
            num_cards_for_hand=5
        )

    # Play a single game
    def play_episode(self, render=False):
        # Reset the chips for every game
        obs = self.dealer.reset(reset_stacks=True)
        # Keep track of who has folded
        done = [False] * self.num_players
        # Initialize proper order for agents
        agent_order = list(self.agents.keys())
        agent_assignments = {i: self.agents[agent_order[i]] for i in range(self.num_players)}
        # Actions taken to update opponent profiles
        actions_taken = [None] * self.num_players

        # Play one episode of the game
        while not all(done) and obs['action'] not in [-1, None]:
            # Get the current player
            current_player_idx = obs['action']

            # Get the current agent to make proper function calls
            agent = agent_assignments[current_player_idx]
            # Get the agent's current stack, hole cards, and community variables
            current_stack = obs['stacks'][current_player_idx]
            hole = obs['hole_cards']
            board = obs['community_cards']
            pot_size = obs['pot']
            min_raise = obs['min_raise']
            street_commits = tuple(obs['street_commits'])

            street = ['preflop', 'flop', 'turn', 'river'][len(board) - 0 if len(board) == 0 else len(board) - 2]

            try:
                if isinstance(agent, MCCFR_N_Player_Complex):
                    # Go through all the previous moves that have been taken each agent to update tendencies
                    for idx, action in enumerate(actions_taken):
                        if idx != current_player_idx and action is not None:
                            agent.update_opponent_profile(idx, action)

                    # Get the abstract information set
                    info_set = agent.abstract_info_set(obs, hole, board, pot_size, street, street_commits, current_player_idx)

                    # Decide what action to take
                    action = agent.sample_action(info_set)
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        # Take a more sophisticated betting strategy, defaulting to the maximum bet size if unavailable
                        max_possible_raise = min(obs.get('max_raise', current_stack), current_stack)
                        bet = agent.determine_bet_size(pot_size, obs['min_raise'], current_stack)
                        bet = min(bet, max_possible_raise)

                elif isinstance(agent, MCCFR_N_Player_Optimized_Bet):
                    info_set = agent.abstract_info_set(obs, hole, board, pot_size, street, street_commits, current_player_idx)
                    action = agent.select_action(info_set)
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        # Get hand strength and use this and street to calculate a bet
                        strength = self.evaluator.evaluate(hole, board)
                        bet = agent.determine_bet_size(strength, pot_size, min_raise, current_stack, street)

                elif isinstance(agent, MCCFR_N_Player_Optimized_Reg):
                    info_set = agent.abstract_info_set(obs, hole, board, pot_size, street, street_commits, current_player_idx)
                    action = agent.select_action(info_set)
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        # More default betting algorithm
                        bet = agent.determine_bet_size(pot_size, min_raise, current_stack)

                elif isinstance(agent, CFRNPlayerAgent):
                    info_set = agent.abstract_info_set(obs, current_player_idx)
                    action = agent.select_action(info_set)
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        bet = min(current_stack, max(obs['min_raise'], obs['pot'] // 2))

                elif isinstance(agent, (RandomAgent, AggressiveAgent)):
                    action = agent.select_action()
                    if action == 'fold':
                        bet = -1
                    elif action == 'call':
                        bet = min(obs['call'], current_stack)
                    else:
                        if isinstance(agent, AggressiveAgent):
                            bet = 0.75 * pot_size
                        else:
                            bet = min(current_stack, max(min_raise, pot_size // 2))

                # Record the action index taken by that agent specifically
                action_idx = agent.actions.index(action)
                actions_taken[current_player_idx] = action_idx

                # Advance the game
                obs, rewards, done = self.dealer.step(bet)

                # OPTIONAL: Show GUI for it
                if render:
                    self.dealer.render(mode='ascii', show_all_hole_cards=True, highlight_winner=True)
                    time.sleep(0.5)

            # If any of the steps fail (happens in some cases), reset the dealer for the episode and try again
            except Exception as e:
                print(f'Error: {e}')
                self.init_dealer()
                return None

        # Return the rewards for each player at the end of the game
        return rewards

    def simulate_games(self, episodes=10000):
        total_rewards = defaultdict(float)
        completed_episodes = 0
        agent_names = list(self.agents.keys())

        # Play each episode
        for episode in range(1, episodes + 1):
            rewards = self.play_episode()
            # If the episode failed, move on
            if rewards is None:
                continue
                
            # Keep track of how many completed episodes we had
            completed_episodes += 1
            for i, name in enumerate(agent_names):
                # Assign proper rewards to each agent
                total_rewards[name] += rewards[i]

            # Track average every 100 episodes
            if completed_episodes % 100 == 0:
                for name in agent_names:
                    avg = total_rewards[name] / completed_episodes
                    self.reward_history[name].append(avg)
                # Give progress report for testing
                print(f"Completed {completed_episodes} valid episodes out of {episode} attempts")

        # Output final results for each agent
        print("\nFinal Results:")
        avg_rewards = {}
        for name in agent_names:
            avg_rewards[name] = total_rewards[name] / completed_episodes if completed_episodes > 0 else 0.0
            print(f"{name}: {avg_rewards[name]:.2f} (Total: {total_rewards[name]:.2f})")

        return avg_rewards

    # Plot the rewards over time for each agent
    def plot_average_rewards(self):
        plt.figure(figsize=(12, 6))
        for name, rewards in self.reward_history.items():
            plt.plot(np.arange(1, len(rewards) + 1) * 100, rewards, label=name)

        plt.title("Average Reward per 100 Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(True)
        plt.show()


# Run simulation and plot
# NOTE: CHANGE THIS TO INCREASE OR DECREASE THE NUMBER OF PLAYERS
n = 4
simulator = NPlayerGameSimulator(num_players=n)
print(f"\nStarting {n}-player simulation...")
avg_rewards = simulator.simulate_games(episodes=20000)
simulator.plot_average_rewards()
