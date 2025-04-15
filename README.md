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

### Testing code individually
Most of the paper talks about the code that is used in the N-Player folder (as the 2 Player code all works in the N Player, and is rather to show the work I did initially).<br><br>
Each file can be run individually to see how agents are trained/evaluated (but this does not provide much insight). In order to get this, the bottom lines of the code may need to be uncommented to show results (specifically in the N-Player folder)<br><br>
The biggest insight comes from running the testing files, where several agents are compared against each other. This can have the number of players (the variable n) adjusted, so long as self.agents has n agents, and any that need to be trained are trained.
They can be arranged in any order, and some can be commented out to reduce the number of players (can recreate similar results by bring n down to 2 and only using 2 agents, but be aware that the code for each agent is slightly different than the 2 Player one).<br><br>
For information on specific policies for each of the agents, see comments or read the paper that goes into more detail on them (specifically those used in the N-Player testing).
