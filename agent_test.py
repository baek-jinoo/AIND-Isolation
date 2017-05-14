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

class IsolationQuartileCheck(unittest.TestCase):
    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(3)

    def test_quadrent_1(self):
        self.assertEqual(1, self.player1.quadrent_check((1, 1)))

    def test_quadrent_2(self):
        self.assertEqual(2, self.player1.quadrent_check((-1, 1)))

    def test_quadrent_3(self):
        self.assertEqual(3, self.player1.quadrent_check((-1, -1)))

    def test_quadrent_4(self):
        self.assertEqual(4, self.player1.quadrent_check((1, -1)))

    def test_quadrent_2_origin(self):
        self.assertEqual(2, self.player1.quadrent_check((0, 0)))

    def test_quadrent_2_x_0_y_positive(self):
        self.assertEqual(2, self.player1.quadrent_check((0, 1)))

    def test_quadrent_1_x_positive_y_0(self):
        self.assertEqual(1, self.player1.quadrent_check((1, 0)))

    def test_quadrent_4_x_0_y_negative(self):
        self.assertEqual(4, self.player1.quadrent_check((0, -1)))

    def test_quadrent_3_x_negative_y_0(self):
        self.assertEqual(3, self.player1.quadrent_check((-1, 0)))

class IsolationFirstTwoMovesRotationTest(unittest.TestCase):
    """
    Unit tests for first two moves rotation to reduce search space
    """

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(3)
        self.player2 = game_agent.AlphaBetaPlayer(3)

    def test_first_two_move_reorientation_in_q1_even_height_width_in_Q4_middle(self):
        game_width = 4
        game_height = 4
        # make one move in Q1
        reoriented_coorindate = self.player1.reorientate_coordinate((2, 2),
                                                                    game_width,
                                                                    game_height)

        #check that method gives back coordinate in Q2
        self.assertEqual((1, 1), reoriented_coorindate)

    def test_first_two_move_reorientation_in_q1_even_height_width_in_Q3(self):
        game_width = 4
        game_height = 4
        # make one move in Q1
        reoriented_coorindate = self.player1.reorientate_coordinate((3, 1),
                                                                    game_width,
                                                                    game_height)

        #check that method gives back coordinate in Q2
        self.assertEqual((1, 0), reoriented_coorindate)

    def test_first_two_move_reorientation_in_q1(self):
        game_width = 3
        game_height = 3
        # make one move in Q1
        reoriented_coorindate = self.player1.reorientate_coordinate((0, 2),
                                                                    game_width,
                                                                    game_height)

        #check that method gives back coordinate in Q2
        self.assertEqual((0, 0), reoriented_coorindate)

    def test_first_two_move_reorientation_in_q2(self):
        game_width = 3
        game_height = 3
        # make one move in Q2
        reoriented_coorindate = self.player1.reorientate_coordinate((0, 0),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in Q2
        self.assertEqual((0, 0), reoriented_coorindate)

    def test_first_two_move_reorientation_in_q3(self):
        game_width = 3
        game_height = 3
        # make one move in Q3
        reoriented_coorindate = self.player1.reorientate_coordinate((2, 0),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in Q2
        self.assertEqual((0, 0), reoriented_coorindate)

    def test_first_two_move_reorientation_in_q4(self):
        game_width = 3
        game_height = 3
        # make one move in Q4
        reoriented_coorindate = self.player1.reorientate_coordinate((2, 2),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in Q2
        self.assertEqual((0, 0), reoriented_coorindate)

    def test_first_two_move_reorientation_on_edge_between_Q2_and_Q3(self):
        game_width = 3
        game_height = 3
       # make one move in Q3
        reoriented_coorindate = self.player1.reorientate_coordinate((1, 0),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in Q2
        self.assertEqual((0, 1), reoriented_coorindate)

    def test_first_two_move_reorientation_on_edge_between_Q2_and_Q1(self):
        game_width = 3
        game_height = 3
        # make one move in Q2
        reoriented_coorindate = self.player1.reorientate_coordinate((0, 1),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in Q2
        self.assertEqual((0, 1), reoriented_coorindate)

    def test_first_two_move_reorientation_in_origin(self):
        game_width = 3
        game_height = 3
        # make one move in Q2
        reoriented_coorindate = self.player1.reorientate_coordinate((1, 1),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in origin
        self.assertEqual((1, 1), reoriented_coorindate)

    def test_first_two_move_reorientation_on_edge_between_Q3_and_Q4(self):
        game_width = 3
        game_height = 3
        # make one move in Q2
        reoriented_coorindate = self.player1.reorientate_coordinate((2, 1),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back coordinate in Q2
        self.assertEqual((0, 1), reoriented_coorindate)

    def test_first_move_reorientation_in_assymetric_board(self):
        game_width = 2
        game_height = 3
        self.game = isolation.Board(self.player1, self.player2, game_width,
                                    game_height)
        reoriented_coorindate = self.player1.reorientate_coordinate((2, 1),
                                                                    game_width,
                                                                    game_height)
        #check that method gives back the same coordinates
        self.assertEqual((2, 1), reoriented_coorindate)

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
    def test_forefeit(self):
        moves = [[0, 4]]
        #moves = [[0, 4], [2, 6], [2, 3], [4, 5], [1, 5], [5, 3], [3, 4], [6, 5],
        #         [5, 5], [4, 4], [4, 3], [2, 5], [6, 4], [3, 3], [5, 6], [5, 2],
        #         [3, 5], [3, 1], [1, 4], [1, 2], [0, 2], [2, 4], [1, 0], [0, 3],
        #         [2, 2]]
        for move in moves:
            self.game.apply_move(move)

        print(self.game.to_string())
    #    time_millis = lambda: 1000 * timeit.default_timer()
    #    move_start = time_millis()
    #    time_left = lambda : 150 - (time_millis() - move_start)
    #    next_move = self.player2.get_move(self.game, time_left)
    #    print(next_move)

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
