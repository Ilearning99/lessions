#!/usr/bin/env python3
"""This program plays a game of Rock, Paper, Scissors between two Players,
and reports both Player's scores each round."""
import random

moves = ['rock', 'paper', 'scissors']

"""The Player class is the parent class for all of the Players
in this game"""


class Player:
    def move(self):
        return 'rock'

    def learn(self, my_move, their_move):
        pass


class RandomPlayer(Player):
    def move(self):
        return random.choice(moves)


class HumanPlayer(Player):
    def move(self):
        while True:
            next_move = input("Rock, paper, scissors? > ")
            next_move = next_move.lower()
            if next_move in moves:
                return next_move


class ReflectPlayer(Player):
    pre_move = "rock"

    def move(self):
        return pre_move

    def learn(self, my_move, their_move):
        pre_move = their_move


class CyclePlayer(Player):
    pre_move = 0

    def move(self):
        pre_move += 1
        pre_move %= 3
        return moves[pre_move]


def beats(one, two):
    return ((one == 'rock' and two == 'scissors') or
            (one == 'scissors' and two == 'paper') or
            (one == 'paper' and two == 'rock'))


class Game:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.score = [0, 0]

    def play_round(self):
        move1 = self.p1.move()
        move2 = self.p2.move()
        print(f"You played {move1}.\n Opponent played {move2}.")
        if move1 == move2:
            print("This is a tie!")
        elif beats(move1, move2):
            print("You win this time!")
            self.score[0] += 1
        else:
            print("You loss this time!")
            self.score[1] += 1
        print(f"Score: You {self.score[0]}, The opponent {self.score[1]}")
        self.p1.learn(move1, move2)
        self.p2.learn(move2, move1)

    def play_game(self):
        print("Game start!")
        round = 0
        while True:
            order = input("do you want to quit:")
            if order == 'quit':
                break
            print(f"Round {round}:")
            self.play_round()
        print("Game over!")
        if self.score[0] > self.score[1]:
            print(f"You win, your score is {self.score[0]}")
        elif self.score[0] < self.score[1]:
            print(f"You loss, your score is {self.score[0]}")
        else:
            print(f"It is a tie, your score is {self.score[0]}")


if __name__ == '__main__':
    game = Game(HumanPlayer(), RandomPlayer())
    game.play_game()
