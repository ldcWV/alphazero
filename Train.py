import os
import torch
from Connect4.NeuralNetwork import *
from Connect4.Game import *
import math
import numpy as np
import copy
from Referee import Referee
import torch.optim as optim
from tqdm import tqdm
import random
import wandb

N = 1000
GAMES_PER_ITERATION = 100
C_PUCT = 1

class MCTS:
    class Node:
        def __init__(self, policy):
            self.q_value = [0.0 for i in range(GameEngine.NUM_ACTIONS)]
            self.num_visits = [0 for i in range(GameEngine.NUM_ACTIONS)]
            self.policy = policy
    
    def mctsRec(self, nn, engine, state): # returns the value at leaf from perspective of other player
        # Has game ended?
        if engine.isGameOver(state):
            return -engine.getOutcome(state) * engine.getTurn(state)
        
        stateAsInt = state.toInt()
        
        # Have we been to this node?
        if stateAsInt in self.visited:
            # Pick action with highest u value
            tot_visits = 1 + sum(self.nodes[stateAsInt].num_visits) # add 1 to account for first visit
            
            actions = engine.getLegalActions(state)
            best_u = -1000000
            best_action = actions[0]
            for action in actions:
                node = self.nodes[stateAsInt]
                u = node.q_value[action] + C_PUCT * node.policy[action] * math.sqrt(tot_visits) / (1 + node.num_visits[action])
                if u > best_u:
                    best_u = u
                    best_action = action
            
            # Recurse on resulting state
            v = self.mctsRec(nn, engine, engine.getNextState(state, best_action))
            
            # Update q value and number of visits
            node = self.nodes[stateAsInt]
            w = node.num_visits[best_action] / (node.num_visits[best_action] + 1)
            node.q_value[best_action] = node.q_value[best_action]*w + v*(1-w)
            node.num_visits[best_action] += 1
            
            return -v
        else:
            self.visited.add(stateAsInt)
            inp = state.toNeuralNetworkInput().to(next(nn.parameters()).device)
            pi, val = nn(inp)
            pi = torch.exp(pi)
            self.nodes[stateAsInt] = self.Node(pi.view(-1))
            return -val
    
    def searchTurn(self, nn, engine, state, simulations_per_turn): # returns the improved policy
        # Build MCTS nodes
        self.nodes = dict()
        self.visited = set()
        for i in range(simulations_per_turn):
            self.mctsRec(nn, engine, state)
        root_node = self.nodes[state.toInt()]
        
        # Compute improved policy
        improved_policy = [0 for i in range(GameEngine.NUM_ACTIONS)]
        for action in engine.getLegalActions(state):
            improved_policy[action] = root_node.num_visits[action]
        tot = sum(improved_policy)
        for i in range(GameEngine.NUM_ACTIONS):
            improved_policy[i] /= tot
        return improved_policy

    def selfPlay(self, nn, engine, simulations_per_turn=25): # returns list of (state, policy, final outcome)
        game_history = []
        state = State()
        while(True):
            if engine.isGameOver(state):
                # last_value is value of last state in game_history
                last_value = -engine.getOutcome(state) * engine.getTurn(state)
                break
            
            # Compute the new policy by MCTS searches
            improved_policy = self.searchTurn(nn, engine, state, simulations_per_turn)
            game_history.append((state, improved_policy))
            
            # Add noise to the new policy
            noise = np.reshape(np.random.dirichlet([0.03 for i in range(GameEngine.NUM_ACTIONS)], 1), (-1))
            eps = 0.25
            pi = np.array(improved_policy)
            pi = pi*(1-eps) + noise*eps
            
            # Pick an action by randomly sampling policy
            probs = []
            actions = []
            for i in engine.getLegalActions(state):
                probs.append(pi[i])
                actions.append(i)
            tot = sum(probs)
            for i in range(len(probs)):
                probs[i] /= tot
            action = np.random.choice(actions, p=probs)
            
            # Update state
            state = engine.getNextState(state, action)
        
        res = []
        cur_value = last_value
        for i in reversed(range(0, len(game_history))):
            res.append((game_history[i][0], game_history[i][1], cur_value))
            cur_value *= -1
        res.reverse()
        return res

def getNNPlayer(nn):
    def move(state):
        mcts = MCTS()
        engine = GameEngine()
        policy = mcts.searchTurn(nn, engine, state, 25)
        
        pi = np.array(policy)
        noise = np.reshape(np.random.dirichlet([0.03 for i in range(GameEngine.NUM_ACTIONS)], 1), (-1))
        eps = 0.1
        pi = pi*(1-eps) + noise*eps
        
        probs = []
        actions = []
        for i in engine.getLegalActions(state):
            probs.append(pi[i])
            actions.append(i)
        action = actions[np.argmax(probs)]
        
        return action
    return move

def train(nn, data, num_epochs=30):
    dataset = Connect4Dataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    optimizer = optim.Adam(nn.parameters())

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        for (state, policy, value) in dataloader:
            state = state.to(device)
            policy = policy.to(device)
            value = value.to(device)
            
            out_p, out_v = nn(state)
            
            batch_size = state.shape[0]
            loss_p = -torch.sum(policy * out_p) / batch_size
            loss_v = torch.sum((value - out_v) ** 2) / batch_size
            loss = loss_p + loss_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wandb.log({
                'epoch': epoch,
                'loss': loss
            })
    
if __name__ == "__main__":    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    engine = GameEngine()
    
    nn = NeuralNetwork()
    nn.eval()
    nn.to(device)
    
    best_nn = copy.deepcopy(nn)
    
    initial_nn = copy.deepcopy(nn)
    mcts = MCTS()
    referee = Referee()
    
    state1 = State()
    state1.board = [
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1,-1, 0],
        [ 0, 0, 0, 1,-1, 1, 0],
        [ 0, 1, 1,-1,-1,-1, 0]
    ]
    state2 = State()
    state2.board = [
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,-1,-1, 0, 0, 0],
        [-1, 1, 1, 1, 0, 0, 0]
    ]
    
    wandb.init(project="alphazero-connect4")

    for i in tqdm(range(N), desc="outer loop"):
        policy, value = nn(state1.toNeuralNetworkInput().to(device))
        policy = torch.exp(policy)
        print(state1)
        print(policy, value)
        policy, value = nn(state2.toNeuralNetworkInput().to(device))
        policy = torch.exp(policy)
        print(state2)
        print(policy, value)
        
        samples = []
        for j in tqdm(range(GAMES_PER_ITERATION), desc="self play"):
            samples += mcts.selfPlay(nn, engine)
        
        tqdm.write("Training on self play results")
        nn.train()
        train(nn, samples)
        nn.eval()
        
        state = samples[-1][0]
        policy, value = nn(state.toNeuralNetworkInput().to(device))
        policy = torch.exp(policy)
        print(state)
        print("expected policy:")
        print(samples[-1][1])
        print("got policy:")
        print(policy)
        print("expected value:")
        print(samples[-1][2])
        print("got value:")
        print(value)
        
        torch.save(nn.state_dict(), f"models/nn_{i}.pt")
        
        tqdm.write("Benchmarking against best so far")
        win_rate, draw_rate, loss_rate = referee.calcWinRate(getNNPlayer(nn), getNNPlayer(best_nn), 50)
        
        tqdm.write(f"WDL against best so far: {(win_rate, draw_rate, loss_rate)}")
        wandb.log({
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate
        })
        if win_rate + draw_rate >= 0.51:
            tqdm.write("Replacing best version")
            best_nn = copy.deepcopy(nn)
        else:
            tqdm.write("Best version not replaced; resetting current to best")
            nn = copy.deepcopy(best_nn)
        
        tqdm.write("Playing against initial")
        wr2 = referee.calcWinRate(getNNPlayer(nn), getNNPlayer(initial_nn), 20)
        wandb.log({
            'win_rate_2': wr2[0],
            'draw_rate_2': wr2[1],
            'loss_rate_2': wr2[2]
        })
        tqdm.write(f"WDL against initial = {wr2}")
        
        tqdm.write("--------------------------------")
