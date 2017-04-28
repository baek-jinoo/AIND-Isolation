"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import random
import timeit

from importlib import reload


class IsolationMinimaxTest(unittest.TestCase):
    """Unit tests for minimax isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer(4)
        self.player2 = game_agent.MinimaxPlayer(3)
        self.game = isolation.Board(self.player1, self.player2, 7, 7)

    @unittest.skip
    def test_minmax_get_move(self):
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda : 30000 - (time_millis() - move_start)
        self.player1.get_move(self.game, time_left)

    @unittest.skip
    def test_minimax(self):
        print("start minimax test")
        (winner, history, reason) = self.game.play()
        print(winner.name)
        self.assertEqual(self.player2, winner)
        self.assertEqual(self.player1, winner)

    @unittest.skip
    def test_custom_score(self):
        #print(game_agent.custom_score(self.game, self.player1))
        pass

class IsolationAlphaBetaTest(unittest.TestCase):
    """Unit tests for alpha beta isolation agents"""
    
    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(10)
        self.player2 = game_agent.AlphaBetaPlayer(10)
        self.game = isolation.Board(self.player1, self.player2, 7, 7)

    @unittest.skip
    def test_alpha_beta_pruning_get_move(self):
        print("start alpha beta pruning test")
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda : 30000 - (time_millis() - move_start)
        self.player1.get_move(self.game, time_left)

    #@unittest.skip
    def test_alpha_beta_pruning(self):
        (winner, history, reason) = self.game.play()
        print(winner.name)
        self.assertEqual(self.player2, winner)
        self.assertEqual(self.player1, winner)

if __name__ == '__main__':
    unittest.main()
