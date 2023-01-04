from Connect4.Game import *

class Referee:
    def __init__(self):
        self.engine = GameEngine()
    
    def runGame(self, player1, player2):
        state = State()
        turn = 1
        while True:
            if self.engine.isGameOver(state):
                return self.engine.getOutcome(state)
            if turn == 1:
                state = self.engine.getNextState(state, player1(state))
            else:
                state = self.engine.getNextState(state, player2(state))
            turn *= -1
    
    def calcWinRate(self, player1, player2, numGames=100):
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(numGames):
            result = self.runGame(player1, player2) if i%2 == 0 else -self.runGame(player2, player1)
            if result == 1:
                wins += 1
            elif result == 0:
                draws = 1
            else:
                losses += 1
        
        return wins/numGames, draws/numGames, losses/numGames
