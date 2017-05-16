"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def half_width_and_height(game):
    """
    return half of the width and height of the game board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    -------
    (float, float)
        (half of game width, half of game height)
    """
    return game.width / 2., game.height / 2.

def center_score(game, player):
    """
    Return center score of the player. The manhattan distance from the center to
    the player

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        Return the manhattan distance from the center to the player. zero if the
        player is not on the board
    """
    w, h = half_width_and_height(game)
    location = game.get_player_location(player)
    if location != None:
        return float(abs(h - location[0]) + abs(w - location[1]))
    return 0.

def center_score_max(game):
    """
    Return the maximum manhattan distance

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    -------
    float
        Return the maximum manhattan distance from the center
    """
    w, h = half_width_and_height(game)
    return float(w + h)

def number_of_open_moves(game, player):
    """
    Return the number of open moves between the current player 

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        Return the number of open moves between the current player 
    """
    loc = game.get_player_location(player)
    if loc == None:
        return self.get_blank_spaces()

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]

    r, c = loc
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                   if game.move_is_legal((r + dr, c + dc))]
    return float(len(valid_moves))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    game_utility_value = game.utility(player)
    if game_utility_value != 0.:
        return game_utility_value

    # open move difference
    my_next_legal_moves = number_of_open_moves(game, player)
    their_next_legal_moves = number_of_open_moves(game, game.get_opponent(player))

    open_move_difference = float(my_next_legal_moves - their_next_legal_moves)

    # manhattan distance from center
    current_center_score_max = center_score_max(game)
    my_center_score = center_score(game, player) / current_center_score_max

    # manhattan distance from opponent
    r, c = game.get_player_location(player)
    their_r, their_c = game.get_player_location(game.get_opponent(player))
    distance_from_opponent = float(abs(r - their_r) + abs(c - their_c)) / float(game.width + game.height)

    #calculate 7x7 cluster around me only when blank spaces are 30% filled up
    blank_spaces = float(len(game.get_blank_spaces()))
    total_spaces = float(game.width * game.height)
    if blank_spaces / total_spaces < 0.7:
        return open_move_difference + distance_from_opponent + my_center_score 

    surrounding_nodes = []
    for y in range(-3, 4):
        for x in range(-3, 4):
            if y == 0 and x == 0:
                continue
            surrounding_nodes.append((y, x))

    taken_nodes = [(r + dr, c + dc) for dr, dc in surrounding_nodes
                   if not game.move_is_legal((r + dr, c + dc))]

    return open_move_difference + distance_from_opponent + my_center_score + float(len(taken_nodes))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    game_utility_value = game.utility(player)
    if game_utility_value != 0.:
        return game_utility_value

    # open move difference
    my_next_legal_moves = number_of_open_moves(game, player)
    their_next_legal_moves = number_of_open_moves(game, game.get_opponent(player))

    open_move_difference = float(my_next_legal_moves - their_next_legal_moves)

    # manhattan distance from opponent
    r, c = game.get_player_location(player)
    their_r, their_c = game.get_player_location(game.get_opponent(player))
    distance_from_opponent = float(abs(r - their_r) + abs(c - their_c)) / float(game.width + game.height)

    # manhattan distance from center
    current_center_score_max = center_score_max(game)
    my_center_score = center_score(game, player) / current_center_score_max

    return open_move_difference + distance_from_opponent + my_center_score


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    game_utility_value = game.utility(player)
    if game_utility_value != 0.:
        return game_utility_value

    # open move difference
    my_next_legal_moves = number_of_open_moves(game, player)
    their_next_legal_moves = number_of_open_moves(game, game.get_opponent(player))

    open_move_difference = float(my_next_legal_moves - their_next_legal_moves)

    # center score
    current_center_score_max = center_score_max(game)

    my_center_score = center_score(game, player) / current_center_score_max
    return open_move_difference + my_center_score

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.debug = False
        self.checked_nodes = []

    def reorientate_coordinate(self, coordinate, game_width, game_height):
        """Return the coordinate (y, x) that would be equivalent closer to the origin
        if the board was symmetrically rotated. This will only do this if the
        board is a square. E.g. on a 3 x 3 board, input of (0, 2) and (2, 2) would return
        (0, 0)

        Parameters
        ----------
        coordinate : (int, int)
            y, x coordinate on the board where (0, 0) is top left and increases
            to the right and down
        game_width : int
            the width of the board
        game_height : int
            the height of the board
        Returns
        -------
        int
            Return the quadrent number in the cartesian coordinate system
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # only do this for symmetric board
        if game_height != game_width:
            return coordinate

        # reverse order because input is (y, x) and we work in (x, y)
        coordinate = coordinate[::-1]

        x_in = coordinate[0]
        flipped_y_in = coordinate[1]
        # flip y because we want to work in euclidean coordinates
        y_in = (game_height - 1) - flipped_y_in
        
        # transform to make the coordinates be symmetrical
        euclidean_x = 2 * x_in - (game_width - 1)
        euclidean_y = 2 * y_in - (game_height - 1)
        euclidean_coordinate = (euclidean_x, euclidean_y)

        # do quartile check
        transformation = numpy.matrix([[1., 0.], [0., 1.]])
        quadrent = self.quadrent_check(euclidean_coordinate)
        if quadrent == 1:
            transformation = numpy.matrix([[0., -1.], [1., 0.]])
        if quadrent == 3:
            transformation = numpy.matrix([[0., 1.], [-1., 0.]])
        if quadrent == 4:
            transformation = numpy.matrix([[-1., 0.], [0., -1.]])


        # make numpy vector from tuple
        coordinate_vector = numpy.transpose(numpy.matrix(euclidean_coordinate))


        # linear transformation
        transformed_matrix = numpy.transpose(transformation * coordinate_vector)
        transformed_array = numpy.squeeze(numpy.asarray(transformed_matrix))

        transformed_euclidean_x = transformed_array[0]
        transformed_euclidean_y = transformed_array[1]

        # undo the board transform to undo the symmetry
        transformed_x = int((transformed_euclidean_x + game_width - 1) * 0.5)
        flipped_transformed_y = int((transformed_euclidean_y + game_height - 1) * 0.5)
        # unflip the y coordinate
        transformed_y = (game_height - 1) - flipped_transformed_y

        # return as (y, x)
        return (transformed_y, transformed_x)

    def quadrent_check(self, cartesian_coordinate):
        """Return the quadrant in a cartesian coordinate system

        Parameters
        ----------
        cartesian_coordinate : (int, int)
            x, y coordinate in the carteisan coordinate system

        Returns
        -------
        int
            Return the quadrent number in the cartesian coordinate system
        """
        x = cartesian_coordinate[0]
        y = cartesian_coordinate[1]
        if x == 0 and y == 0:
            return 2

        if x == 0 or y == 0:
            if y > 0:
                return 2
            if y < 0:
                return 4
            if x > 0:
                return 1
            if x < 0:
                return 3

        if x > 0:
            if y > 0:
                return 1
            else:
                return 4
        else:
            if y > 0:
                return 2
            else:
                return 3

    def deduplicated_symmetrical_legal_moves(self, game, legal_moves):
        """If the current active player has not moved yet, we will remove the
        symmetrical moves to reduce the branching factor.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        legal_moves : List
            List of legal moves for the current active player

        Returns
        -------
        List
            Return the list of legal moves. This is a shorter list when compared
            to the input if the board is a square and there are symmetrical
            moves for the user
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        active_player_location = game.get_player_location(game.active_player)

        if active_player_location != None:
            return legal_moves

        opponent_player = game.get_opponent(game.active_player)
        opponent_player_location = game.get_player_location(opponent_player)
        if opponent_player_location != None:
            center_move_exists = game.width % 2 == 1 and game.height % 2 == 1 and game.width == game.height
            if not center_move_exists:
                return legal_moves

            center_move = (int(game.height / 2), int(game.width / 2))
            if center_move != opponent_player_location:
                return legal_moves

        legal_moves_set = set()
        for legal_move in legal_moves:
            reoriented_coordinate = self.reorientate_coordinate(legal_move,
                                                                game.width,
                                                                game.height)
            legal_moves_set.add(reoriented_coordinate)
        return list(legal_moves_set)

    def terminal_test(self, game):
        """Check if the game has ended

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        Returns
        -------
        bool
            return whether there are legal move left for the current active
            player
        """
        if not game.get_legal_moves(game.active_player):
            return True
        return False

    def legal_moves_and_random_move(self, game):
        """Get a list of legal moves and a random choice from the list

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        Returns
        -------
        (List, (int, int))
            return the list of legal moves and one of its content randomly
        chosen. If there is nothing in the list, it will return (-1, -1)
        """
        legal_moves = game.get_legal_moves(game.active_player)
        move = (-1 , -1)
        if len(legal_moves) > 0:
            move = random.choice(legal_moves)
        return (legal_moves, move)


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        (legal_moves, move) = self.legal_moves_and_random_move(game)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        value = None
        (legal_moves, next_move) = self.legal_moves_and_random_move(game)

        if depth <= 0:
            return next_move

        legal_moves = self.deduplicated_symmetrical_legal_moves(game, legal_moves)

        for legal_move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            forecasted_game = game.forecast_move(legal_move)
            if self.debug:
                self.checked_nodes.append(forecasted_game)
            new_value = self.min_value(forecasted_game, depth - 1)
            if value == None or value < new_value:
                value = new_value
                next_move = legal_move
        return next_move

    def min_value(self, game, depth):
        """Get the min value for next move

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : float
            The maximum depth of plies in the gameplay to iterate through the
            depth first search

        Returns
        -------
        float
            the minimum utility value or evaluation function value of the
            current state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0 or self.terminal_test(game):
            return self.score(game, self)
        v = float("inf")
        legal_moves = game.get_legal_moves(game.active_player)
        legal_moves = self.deduplicated_symmetrical_legal_moves(game, legal_moves)

        forecasted_games = [game.forecast_move(legal_move) for legal_move in
         legal_moves]
        for forecasted_game in forecasted_games:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if self.debug:
                self.checked_nodes.append(forecasted_game)
            v = min(v, self.max_value(forecasted_game, depth - 1))
        return v

    def max_value(self, game, depth):
        """Get the max value for next move

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : float
            The maximum depth of plies in the gameplay to iterate through the
            depth first search

        Returns
        -------
        float
            the maximum utility value or evaluation function value of the 
            current state
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0 or self.terminal_test(game):
            return self.score(game, self)
        v = float("-inf")

        forecasted_games = [game.forecast_move(legal_move) for legal_move in
         game.get_legal_moves(game.active_player)]
        for forecasted_game in forecasted_games:
            if self.debug:
                self.checked_nodes.append(forecasted_game)
            v = max(v, self.min_value(forecasted_game, depth - 1))
        return v

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        (legal_moves, move) = self.legal_moves_and_random_move(game)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            depth = 1

            while True:
                move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            return move

        # Return the best move from the last completed search iteration
        return move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if depth == 0:
            (legal_moves, move) = self.legal_moves_and_random_move(game)
            return move

        (move, new_value) = self.max_value(game, depth, alpha, beta)
        return move

    def min_value(self, game, depth, alpha, beta):
        """Get the min value for next move. For the first two moves, we remove
        symmetrical moves to reduce the branching factor

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : float
            The maximum depth of plies in the gameplay to iterate through the
            depth first search
        alpha: float
            The value used to cut-off serach on the min node. This value is
            updated in a max node and used in the subsequent sibling max node
        beta: float
            The value used to cut-off search under the max node. This value is
            updated in a min node and used in the subsequent sibling min node

        Returns
        -------
        ((int, int), float)
            (best_move, utility_value)
            The best move in the board coordinate
            and the maximum utility value or evaluation function
            value of the current state
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        (legal_moves, move) = self.legal_moves_and_random_move(game)

        if depth <= 0 or self.terminal_test(game):
            return (move, self.score(game, self))
        v = float("inf")

        legal_moves = self.deduplicated_symmetrical_legal_moves(game, legal_moves)

        for legal_move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            forecasted_game = game.forecast_move(legal_move)
            if self.debug == True:
                self.checked_nodes.append(forecasted_game)
            (returned_move, new_value) = self.max_value(forecasted_game, depth - 1, alpha, beta)
            if v > new_value:
                v = new_value
                move = legal_move
            if v <= alpha:
                return (move, v)
            beta = min(beta, v)
        return (move, v)

    def max_value(self, game, depth, alpha, beta):
        """Get the max value for next move. For the first two moves, we remove
        symmetrical moves to reduce the branching factor

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : float
            The maximum depth of plies in the gameplay to iterate through the
            depth first search
        alpha: float
            The value used to cut-off serach on the min node. This value is
            updated in a max node and used in the subsequent sibling max node
        beta: float
            The value used to cut-off search under the max node. This value is
            updated in a min node and used in the subsequent sibling min node

        Returns
        -------
        ((int, int), float)
            (best_move, utility_value)
            The best move in the board coordinate
            and the maximum utility value or evaluation function
            value of the current state
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        (legal_moves, move) = self.legal_moves_and_random_move(game)

        if depth <= 0 or self.terminal_test(game):
            return (move, self.score(game, self))
        v = float("-inf")

        legal_moves = self.deduplicated_symmetrical_legal_moves(game, legal_moves)

        for legal_move in legal_moves:
            forecasted_game = game.forecast_move(legal_move)
            if self.debug == True:
                self.checked_nodes.append(forecasted_game)
            (returned_move, new_value) = self.min_value(forecasted_game, depth - 1, alpha, beta)
            if v < new_value:
                v = new_value
                move = legal_move
            if v >= beta:
                return (move, v)
            alpha = max(alpha, v)
        return (move, v)

