
"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys

def heuristic_score_simple(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def heuristic_score_moves_to_board(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    board_size = game.height * game.width
    board_size = game.height * game.width
    moves_to_board = game.move_count / board_size
    return float((own_moves*moves_to_board*2 - opp_moves))

def heuristic_score_weighted(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves * 2 - opp_moves)

def heuristic_score_weighted_with_board(game, player):
    blank_spaces = len(game.get_blank_spaces())
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves * 3 - opp_moves * 2 + blank_spaces * 1)

def heuristic_score_weighted_with_board_defensive_to_offensive(game, player):
    blank_spaces = len(game.get_blank_spaces())
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    board_size = game.height * game.width
    moves_to_board = game.move_count / board_size
    
    if moves_to_board <= 0.5:
        return float(own_moves * 2 - opp_moves)
    else:
        return float(own_moves - opp_moves * 2)
    
def heuristic_score_weighted_with_board_offensive_to_defensive(game, player):
    blank_spaces = len(game.get_blank_spaces())
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    board_size = game.height * game.width
    moves_to_board = game.move_count / board_size
    
    if moves_to_board > 0.5:
        return float(own_moves * 2 - opp_moves)
    else:
        return float(own_moves - opp_moves * 2)
    
def heuristic_score_block_opponent(game, player):
    play_moves = game.get_legal_moves(player) 
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    same_moves = len(list(set(play_moves) & set(opp_moves)))
    board_size = game.height * game.width
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    moves_to_board = game.move_count / board_size
    
    if moves_to_board <= 0.33:
        return float(own_moves * 2 - opp_moves)
    elif moves_to_board > 0.33 and moves_to_board <= 0.66:
        return float(own_moves * 2 - opp_moves )
    else:
        return float(own_moves - opp_moves * 2 + same_moves)

heuristic = {
    'heuristic_score_simple': heuristic_score_simple,
    'heuristic_score_moves_to_board': heuristic_score_moves_to_board,
    'heuristic_score_weighted': heuristic_score_weighted,
    'heuristic_score_weighted_with_board': heuristic_score_weighted_with_board,
    'heuristic_score_weighted_with_board_defensive_to_offensive': heuristic_score_weighted_with_board_defensive_to_offensive,
    'heuristic_score_weighted_with_board_offensive_to_defensive': heuristic_score_weighted_with_board_offensive_to_defensive,
    'heuristic_score_block_opponent': heuristic_score_block_opponent
}

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def custom_score(game, player):
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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return heuristic[sys.argv[1]]


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.POS_INF = float("inf")
        self.NEG_INF = float("-inf")
        self.suicide_move = (-1, -1)
    
    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

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

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves: 
            return self.suicide_move

        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        best_score = self.NEG_INF

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method is 'minimax':
                search = self.minimax
            else:
                search = self.alphabeta

            if self.iterative:
                search_depth = 1
                while 1:
                    
                    score, next_move = search(game, depth=search_depth, maximizing_player=True)
                    #check if new score is better than previous
                    if (score, next_move) > (best_score, best_move):
                        (best_score, best_move) = (score, next_move)
                        
                    #increase the search depth
                    search_depth += 1
            else:
                score, next_move = search(game, self.search_depth)

                if (score, next_move) > (best_score, best_move):
                    (best_score, best_move) = (score, next_move)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        return best_move

        # Return the best move from the last completed search iteration

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        best_move = self.suicide_move
        
        #assign the best score based on if the player is a maximizing player or minimizing player
        best_score = self.NEG_INF if maximizing_player else self.POS_INF
        optimizer = max if maximizing_player else min
        
        if self.time_left() < self.TIMER_THRESHOLD: raise Timeout()
        if depth is 0: return self.score(game, self), best_move
        
        for move in game.get_legal_moves():
            #recursively call minimax on next move
            score, _ = self.minimax(game.forecast_move(move), depth - 1, not maximizing_player)
            best_score, best_move = optimizer((best_score, best_move), (score, move))
            
        return best_score, best_move
    
    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

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

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
            
        best_move = self.suicide_move
        best_score = alpha if maximizing_player else beta
        
        if self.time_left() < self.TIMER_THRESHOLD: raise Timeout()
        if depth is 0: return self.score(game, self), best_move
        
        if maximizing_player:
            for move in game.get_legal_moves():
                
                #return score for the move 
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, best_score, beta, not maximizing_player)
                
                if score > best_score: 
                    best_score, best_move = score, move
                if best_score >= beta: 
                    return (best_score, best_move)
        else:
            # Iterate through all possible legale moves
            for move in game.get_legal_moves():
                # Return the score for that 
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, best_score, not maximizing_player)
                
                if score < best_score: 
                    best_score, best_move = score, move
                if best_score <= alpha: 
                    return (best_score, best_move)
                
        return best_score, best_move
