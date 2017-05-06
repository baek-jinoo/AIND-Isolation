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
        self.player1 = game_agent.AlphaBetaPlayer(3)
        self.player2 = game_agent.AlphaBetaPlayer(3)
        self.game = isolation.Board(self.player1, self.player2, 7, 7)

        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda : 30000 - (time_millis() - move_start)
        self.player1.time_left = time_left

    @unittest.skip
    def test_alpha_beta_depth_0(self):
        next_move = self.player1.alphabeta(self.game, 0)

        self.assertEqual(next_move, (-1, -1))

    @unittest.skip
    def test_alpha_beta_depth_1(self):
        expected_checked_nodes = []
        for idx_x in range(0, 3):
            for idx_y in range(0, 3):
                expected_checked_nodes.append(self.game.forecast_move((idx_x, idx_y)).hash())

        next_move = self.player1.alphabeta(self.game, 1)

        self.assertNotEqual(next_move, (-1, -1))

        hashes_of_checked_nodes = [checked_node.hash() for checked_node in self.player1.checked_nodes]

        self.assertEqual(set(expected_checked_nodes), set(hashes_of_checked_nodes))

    @unittest.skip
    def test_alpha_beta_pruning_with_alpha(self):
        next_move = self.player1.alphabeta(self.game, 2)
        #self.assertEqual(36 ,len(self.player1.checked_nodes))

    @unittest.skip
    def test_alpha_beta_pruning_with_beta(self):
        next_move = self.player1.alphabeta(self.game, 3)
        #self.assertEqual(36 ,len(self.player1.checked_nodes))

if __name__ == '__main__':
    unittest.main()
