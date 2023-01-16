import torch
from Connect4.NeuralNetwork import *
from Connect4.Game import *
from Train import getNNPlayer

opponent = "models/nn_46.pt"
model = NeuralNetwork()
model.load_state_dict(torch.load(opponent))
model.eval()
opponent_player = getNNPlayer(model)

engine = GameEngine()
state = State()

turn = 0
while True:
    if engine.isGameOver(state):
        print(state)
        print("Game Over!")
        break

    if turn%2 == 0:
        print(state)
        move = int(input("Enter your move: "))
    else:
        print(state)
        print("AI move...")
        move = opponent_player(state)
    state = engine.getNextState(state, move)
    turn += 1
