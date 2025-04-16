# CS-5100-Poker-CFR
Final CS 5100 Project: A Counterfactual regret minimization approach to creating an optimal Texas Hold'em Poker agent. Utilizes several approaches towards reinforcement learning to achieve a Nash equilibrium policy against several other trained agents.

## Running

### Prerequisites
- Python 3.x

### Installing Libraries
To run this project, you will need the following libraries:  
- numpy
- matplotlib
- clubs (special Poker gymnasium wrapper library utilized in this project)
- time

To install all of these at once, run the following command:

```bash
pip install numpy matplotlib clubs time
```

If this does not work, try making sure you have pip installed, updated to a compatible version, or using pip3 instead

### A Note to the Grader
Nearly all of my paper refers to the N-Player folder implementation agents. I include the 2 Player folder as it provides insight into what I had before I migrated over to N-Player. The 2 player provides many head-to-heads, giving a unique perspective and ability to see how each of the types of agents that were upgraded in the N-Player fare against each other. Should you choose to look at these, they are still thoroughly commented, yet their approaches are slightly off from anything referred to in the paper.

### Testing code individually
Each file can be run individually to see how agents are trained/evaluated (but this does not provide much insight). In order to get this, the bottom lines of the code may need to be uncommented to show results (specifically in the N-Player folder)<br><br>
The biggest insight comes from running the testing files, where several agents are compared against each other. This can have the number of players (the variable n) adjusted, so long as self.agents has n agents, and any that need to be trained are trained.
They can be arranged in any order, and some can be commented out to reduce the number of players (can recreate similar results by bring n down to 2 and only using 2 agents, but be aware that the code for each agent is slightly different than the 2 Player one).<br><br>
For information on specific policies for each of the agents, see comments or read the paper that goes into more detail on them (specifically those used in the N-Player testing).<br><br>
More instructions are listed below on how to run files

### General Notes
1) To get the ASCII GUI: Go into testing_n_player_agents.py and change the render parameter in play_episode to True
2) To run a simulation: adjust the n at the bottom of testing_n_player_agents.py, and make sure that, in the NPlayerGameSimulator class, self.agents is of length n, training any agent that needs training below.
3) To run a single individual agent file, go to the bottom of it and uncomment that last few lines, and just run that file separately (NOTE: This reward meaning does not mean much, however, which is why it is recommended to use testing_n_player_agents.py instead).
