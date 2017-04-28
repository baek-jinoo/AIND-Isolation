"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import sys


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


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
    legal_moves = game.get_legal_moves(player)
    return float(len(legal_moves))

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
    # TODO: finish this function!
    raise NotImplementedError


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
    # TODO: finish this function!
    raise NotImplementedError


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
        random_number = int(100 * random.uniform(0, 1))
        self._name = "player {}".format(random_number)

    @property
    def name(self):
        return self._name

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
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

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
        next_move = (-1, -1)

        if depth <= 0:
            return next_move

        print("starting depth: [{}]".format(depth))
        depth -= 1

        for legal_move in game.get_legal_moves(self):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            forecasted_game = game.forecast_move(legal_move)
            new_value = self.min_value(forecasted_game, depth)
            if value == None or value < new_value:
                print("new value")
                #print(forecasted_game.to_string())
                print("new_value: {}".format(new_value))
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
        #print("new depth: [{}]".format(depth))
        if self.time_left() < self.TIMER_THRESHOLD:
            print("time out")
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0:
            return self.utility_function(game)
        v = float("inf")

        forecasted_games = [game.forecast_move(legal_move) for legal_move in
         game.get_legal_moves(game.active_player)]
        for forecasted_game in forecasted_games:
            #print("min_value loop")
            #print(forecasted_game.to_string())
            before_v = v
            v = min(v, self.max_value(forecasted_game, depth - 1))
            #print("min_value before [{}] after [{}]".format(before_v, v))
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

        #print("new depth: [{}]".format(depth))
        if self.time_left() < self.TIMER_THRESHOLD:
            print("time out")
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0:
            return self.utility_function(game)
        v = float("-inf")

        forecasted_games = [game.forecast_move(legal_move) for legal_move in
         game.get_legal_moves(game.active_player)]
        for forecasted_game in forecasted_games:
            #print("max_value loop")
            #print(forecasted_game.to_string())
            before_v = v
            v = max(v, self.min_value(forecasted_game, depth - 1))
            #print("max_value before [{}] after [{}]".format(before_v, v))
        return v

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

    def utility_function(self, game):
        """Calculate the utility of the game status from the current player's
        perspective

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        Returns
        -------
        float
            float("inf") if the player won
            float("-inf") if the player lost
            the evaluation function if the user did not lose or win
        """

        game_utility_value = game.utility(self)
        if game_utility_value == 0:
            return self.score(game, self)
        return game_utility_value

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

        # TODO implement this properly
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for depth in range(1, self.search_depth + 1):
                print("depth loop: {}".format(depth))
                print("update best move {}".format(best_move))
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            print("caught in exception")
            print("best move when caught in exception: {}".format(best_move))
            pass

        # Return the best move from the last completed search iteration
        return best_move

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
        value = None
        next_move = (-1, -1)

        if depth <= 0:
            return next_move

        #print("starting depth: [{}]".format(depth))
        depth -= 1

        for legal_move in game.get_legal_moves(self):
            print("legal move: {}".format(legal_move))
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            forecasted_game = game.forecast_move(legal_move)
            #print(forecasted_game.to_string())
            #print("new alpha value: {}".format(alpha))
            new_value = self.min_value(forecasted_game, depth, alpha,
                                       beta)
            if value == None or value < new_value:
                #print("new value")
                #print(forecasted_game.to_string())
                #print("new_value: {}".format(new_value))
                value = new_value
                alpha = new_value
                next_move = legal_move
        return next_move

    def min_value(self, game, depth, alpha, beta):
        #TODO update the doc
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
        #print("new depth: [{}]".format(depth))
        if self.time_left() < self.TIMER_THRESHOLD:
            print("time out")
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0:
            return self.utility_function(game)
        v = float("inf")

        forecasted_games = [game.forecast_move(legal_move) for legal_move in
         game.get_legal_moves(game.active_player)]
        for forecasted_game in forecasted_games:
            #print("min_value loop")
            #print(forecasted_game.to_string())
            before_v = v
            v = min(v, self.max_value(forecasted_game, depth - 1, alpha, beta))
            #print("min_value before [{}] after [{}]".format(before_v, v))
            if v <= alpha:
                #print("prune happening in max_value")
                return v
            beta = min(beta, v)
        return v

    def max_value(self, game, depth, alpha, beta):
        #TODO update the doc
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

        #print("new depth: [{}]".format(depth))
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0:
            return self.utility_function(game)
        v = float("-inf")

        forecasted_games = [game.forecast_move(legal_move) for legal_move in
         game.get_legal_moves(game.active_player)]
        for forecasted_game in forecasted_games:
            #print("max_value loop")
            #print(forecasted_game.to_string())
            before_v = v
            v = max(v, self.min_value(forecasted_game, depth - 1, alpha, beta))
            #print("max_value before [{}] after [{}]".format(before_v, v))
            if v >= beta:
                #print("prune happening in  min_value")
                return v
            alpha = max(alpha, v)
        return v

    #TODO DRY this
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

    #TODO DRY this
    def utility_function(self, game):
        """Calculate the utility of the game status from the current player's
        perspective

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        Returns
        -------
        float
            float("inf") if the player won
            float("-inf") if the player lost
            the evaluation function if the user did not lose or win
        """

        game_utility_value = game.utility(self)
        if game_utility_value == 0:
            return self.score(game, self)
        return game_utility_value
