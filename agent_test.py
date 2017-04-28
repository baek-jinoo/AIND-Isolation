"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import random

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer(2)
        self.player2 = game_agent.MinimaxPlayer(2)
        self.game = isolation.Board(self.player1, self.player2, 3, 3)

    def test_bla(self):
        print("start")
        #move = (0, 0)
        #self.game.apply_move(move)
        #print(game_agent.custom_score(self.game, self.player1))
        self.player1.get_move(self.game, 3)
        #(winner, history, reason) = self.game.play()
        #self.assertEqual(self.player2, winner)
        #print(winner)
        #print(history)
        #print(reason)


if __name__ == '__main__':
    unittest.main()
