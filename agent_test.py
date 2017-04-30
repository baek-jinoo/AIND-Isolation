"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest
import timeit

import isolation
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(True, 3)
        self.player2 = game_agent.AlphaBetaPlayer(True, 3)
        self.game = isolation.Board(self.player1, self.player2, 3, 3)

        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda : 30000 - (time_millis() - move_start)
        self.player1.time_left = time_left

    def test_alpha_beta_depth_0(self):
        next_move = self.player1.alphabeta(self.game, 0)

        self.assertEqual(next_move, (-1, -1))

    def test_alpha_beta_depth_1(self):
        next_move = self.player1.alphabeta(self.game, 1)

        self.assertNotEqual(next_move, (-1, -1))
        self.assertEqual(len(self.player1.checked_nodes), 9)
        expected_checked_nodes = []
        for idx in range(0, 3):
            print(idx)
            expected_checked_nodes.append(self.game.forecast_move((0, idx)).hash())
        hashes_of_checked_nodes = [checked_node.hash() for checked_node in self.player1.checked_nodes]
        self.assertEqual(set(expected_checked_nodes), set(hashes_of_checked_nodes))

if __name__ == '__main__':
    unittest.main()
