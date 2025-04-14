import numpy as np
import clubs
import os
import sys
from collections import defaultdict

# Import agents with aliases
from poker_agents import CFRNPlayerAgent as CFR_Agent
from poker_agents import MCCFR_N_Player_Optimized_Bet as MCCFR_Bet_Agent
from poker_agents import MCCFR_N_Player_Optimized_Reg as MCCFR_Reg_Agent
from poker_agents import BasePokerAgent

class PokerSimulator:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.blinds = [1, 2] + [0] * (num_players - 2)
        self.evaluator = clubs.poker.Evaluator(suits=4, ranks=13, cards_for_hand=5)

        # Agent class definitions per type
        self.agent_classes = {
            'CFR': CFR_Agent,
            'MCCFR_Bet': MCCFR_Bet_Agent,
            'MCCFR_Reg': MCCFR_Reg_Agent,
            'Base': BasePokerAgent
        }

        # Initialize agents per type and position
        self.agents = {
            name: [cls(position=i) if 'position' in cls.__init__.__code__.co_varnames
                   else cls() for i in range(num_players)]
            for name, cls in self.agent_classes.items()
        }

        self.init_dealer()

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

    def train_all_agents(self, iterations=5000):
        print("Training all agents...")
        for agent_name in self.agents:
            print(f"\nTraining {agent_name} agents...")
            for position in range(self.num_players):
                print(f"  Position {position}")
                self.train_agent(agent_name, position, iterations)

    def train_agent(self, agent_name, position, iterations):
        agent = self.agents[agent_name][position]

        for it in range(iterations):
            obs = self.dealer.reset(reset_stacks=True)
            done = [False] * self.num_players
            histories = [[] for _ in range(self.num_players)]

            while not all(done) and obs['action'] != -1:
                current_player_idx = obs['action']

                if not obs['active'][current_player_idx]:
                    obs, _, done = self.dealer.step(-1)
                    continue

                # Build info_set & action
                if agent_name == 'CFR':
                    info_set = agent.abstract_info_set(obs, current_player_idx)
                    action = agent.select_action(info_set)
                else:
                    hole = obs['hole_cards']
                    board = obs['community_cards']
                    hand_bucket = agent.get_hand_bucket(hole, board, self.evaluator)
                    stacks = tuple(obs['stacks'])
                    street_commits = tuple(obs['street_commits'])
                    street = ['preflop', 'flop', 'turn', 'river'][len(board) // 2]
                    info_set = (hand_bucket, street, stacks, street_commits)
                    if agent_name == 'MCCFR_Bet':
                        info_set += (current_player_idx,)

                    is_trained_player = (current_player_idx == position)
                    action = agent.select_action(info_set, 1.0, is_trained_player)

                action_idx = agent.actions.index(action)
                bet = agent.get_bet_amount(obs, current_player_idx, action)
                histories[current_player_idx].append((info_set, action_idx))
                obs, rewards, done = self.dealer.step(bet)

            # Regret update
            for info_set, action_idx in histories[position]:
                if agent_name == 'CFR':
                    util = rewards[position]
                    alt_utils = [-util if i != action_idx else util for i in range(3)]
                    agent.update_regrets(info_set, action_idx, util, alt_utils)
                else:
                    strategy = agent.strategy[info_set] if info_set in agent.strategy else np.ones(3) / 3
                    v_s_prime = np.dot(strategy, agent.regrets[info_set])
                    sampling_prob = strategy[action_idx] + 1e-6
                    regret = (rewards[position] - v_s_prime) / sampling_prob
                    agent.update_regrets(info_set, action_idx, regret)

            if (it + 1) % 1000 == 0:
                if hasattr(agent, 'compute_strategy'):
                    agent.compute_strategy()

    def simulate_games(self, episodes=1000):
        results = defaultdict(lambda: np.zeros(self.num_players))

        for agent1_type in self.agents:
            for agent2_type in self.agents:
                if agent1_type == agent2_type:
                    continue

                print(f"\nTesting {agent1_type} vs {agent2_type}...")
                temp_results = np.zeros(self.num_players)

                for _ in range(episodes):
                    obs = self.dealer.reset(reset_stacks=True)
                    done = [False] * self.num_players

                    while not all(done) and obs['action'] != -1:
                        current_player_idx = obs['action']

                        if not obs['active'][current_player_idx]:
                            obs, _, done = self.dealer.step(-1)
                            continue

                        agent_type = agent1_type if current_player_idx % 2 == 0 else agent2_type
                        agent = self.agents[agent_type][current_player_idx]

                        if agent_type == 'CFR':
                            info_set = agent.abstract_info_set(obs, current_player_idx)
                            action = agent.select_action(info_set)
                        else:
                            hole = obs['hole_cards']
                            board = obs['community_cards']
                            hand_bucket = agent.get_hand_bucket(hole, board, self.evaluator)
                            stacks = tuple(obs['stacks'])
                            street_commits = tuple(obs['street_commits'])
                            street = ['preflop', 'flop', 'turn', 'river'][len(board) // 2]
                            info_set = (hand_bucket, street, stacks, street_commits)
                            if agent_type == 'MCCFR_Bet':
                                info_set += (current_player_idx,)

                            action = agent.select_action(info_set, 1.0, True)

                        bet = agent.get_bet_amount(obs, current_player_idx, action)
                        obs, rewards, done = self.dealer.step(bet)

                    temp_results += np.array(rewards)

                avg_results = temp_results / episodes
                print(f"{agent1_type} avg: {avg_results[::2].mean():.2f}")
                print(f"{agent2_type} avg: {avg_results[1::2].mean():.2f}")
                results[(agent1_type, agent2_type)] = avg_results

        return results


if __name__ == "__main__":
    simulator = PokerSimulator(num_players=4)
    simulator.train_all_agents(iterations=5000)
    print("\nStarting simulation tournaments...")
    results = simulator.simulate_games(episodes=1000)

    print("\nFinal Results:")
    for match, rewards in results.items():
        print(f"{match[0]} vs {match[1]}:")
        print(f"  {match[0]} average: {rewards[::2].mean():.2f}")
        print(f"  {match[1]} average: {rewards[1::2].mean():.2f}")
