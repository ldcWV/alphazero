import copy
import torch
import numpy as np

class GameEngine:
    HEIGHT = 6
    WIDTH = 7
    NUM_ACTIONS = WIDTH

    def __init__(self):
        self.state = State()
    
    def isActionLegal(self, state, action):
        return state.board[0][action] == 0
    
    def getLegalActions(self, state):
        res = []
        for action in range(GameEngine.WIDTH):
            if self.isActionLegal(state, action):
                res.append(action)
        return res
    
    def getNextState(self, state, action):
        assert(self.isActionLegal(state, action))
        
        turn = self.getTurn(state)
        
        # Update the board
        res = copy.deepcopy(state)
        j = action
        for i in reversed(range(0, GameEngine.HEIGHT)):
            if state.board[i][j] == 0:
                res.board[i][j] = turn
                break
        return res
    
    def isGameOver(self, state):
        if len(self.getLegalActions(state)) == 0:
            return True
        return self.getOutcome(state) != 0
    
    def getOutcome(self, state):
        for i in range(GameEngine.HEIGHT):
            for j in range(GameEngine.WIDTH - 3):
                if state.board[i][j] != 0 and \
                   state.board[i][j] == state.board[i][j+1] and \
                   state.board[i][j] == state.board[i][j+2] and \
                   state.board[i][j] == state.board[i][j+3]:
                    return state.board[i][j]
        for i in range(GameEngine.HEIGHT - 3):
            for j in range(GameEngine.WIDTH):
                if state.board[i][j] != 0 and \
                   state.board[i][j] == state.board[i+1][j] and \
                   state.board[i][j] == state.board[i+2][j] and \
                   state.board[i][j] == state.board[i+3][j]:
                    return state.board[i][j]
        for i in range(GameEngine.HEIGHT - 3):
            for j in range(GameEngine.WIDTH - 3):
                if state.board[i][j] != 0 and \
                   state.board[i][j] == state.board[i+1][j+1] and \
                   state.board[i][j] == state.board[i+2][j+2] and \
                   state.board[i][j] == state.board[i+3][j+3]:
                    return state.board[i][j]
        for i in range(3, GameEngine.HEIGHT):
            for j in range(GameEngine.WIDTH - 3):
                if state.board[i][j] != 0 and \
                   state.board[i][j] == state.board[i-1][j+1] and \
                   state.board[i][j] == state.board[i-2][j+2] and \
                   state.board[i][j] == state.board[i-3][j+3]:
                    return state.board[i][j]
        return 0
    
    def getTurn(self, state):
        turn = 0
        for i in range(GameEngine.HEIGHT):
            for j in range(GameEngine.WIDTH):
                turn += state.board[i][j]
        if turn == 0:
            turn = 1
        else:
            turn = -1
        return turn

class State:
    def __init__(self):
        self.board = [[0 for j in range(GameEngine.WIDTH)] for i in range(GameEngine.HEIGHT)]
    
    def toNeuralNetworkInput(self):
        board_numpy = np.array(self.board, dtype=np.float32)
        res = np.full((2, GameEngine.HEIGHT, GameEngine.WIDTH), -1.0, dtype=np.float32)
        res[0][board_numpy == -1] = 1.0
        res[1][board_numpy == 1] = 1.0
        return torch.from_numpy(res)
    
    def toInt(self):
        res = 0
        for i in range(GameEngine.HEIGHT):
            for j in range(GameEngine.WIDTH):
                res *= 3
                res += self.board[i][j] + 1
        return res
    
    def __str__(self):
        def to_str(x):
            if x == 0:
                return "_"
            elif x == -1:
                return "O"
            return "X"
        return "\n".join([" ".join([to_str(x) for x in row]) for row in self.board])
