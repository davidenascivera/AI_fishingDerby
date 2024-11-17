# player.py

#!/usr/bin/env python3
import random
import math
import logging

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

# Configure logging
logging.basicConfig(level=logging.INFO, filename='minimax.log', filemode='w',
                    format='%(asctime)s - %(message)s')

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return

class PlayerControllerMinimax(PlayerController):
    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.max_depth = 2  # Adjust depth based on performance

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        """
        # Receive the initial state
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            root = Node(message=msg, player=0)  # Assuming AI is player 0

            # Perform Minimax search to find the best move
            best_move = self.search_best_move(root)

            # Execute the best move
            self.sender({"action": best_move, "search_time": None})

    def search_best_move(self, root: Node) -> str:
        """
        Initiate Minimax search with Alpha-Beta pruning to find the best move.

        :param root: The root node of the current game state.
        :return: Best move as a string ("stay", "left", "right", "up", "down")
        """
        best_val = -float('inf')
        best_move = "stay"

        # Expand children
        children = root.compute_and_get_children()

        logging.info("Starting Minimax search...")

        for child in children:
            move = ACTION_TO_STR.get(child.move, "stay")
            move_val = self.minimax(child, self.max_depth - 1, False, -float('inf'), float('inf'))
            logging.debug(f"Move: {move}, Value: {move_val}")

            if move_val > best_val:
                best_val = move_val
                best_move = move

        logging.info(f"Best Move Selected: {best_move} with Value: {best_val}")
        return best_move

    def minimax(self, node: Node, depth: int, is_maximizing: bool, alpha: float, beta: float) -> float:
        """
        Minimax algorithm with Alpha-Beta pruning.

        :param node: Current game tree node.
        :param depth: Current depth in the tree.
        :param is_maximizing: Boolean indicating if the current layer is maximizing.
        :param alpha: Alpha value for pruning.
        :param beta: Beta value for pruning.
        :return: Heuristic value of the node.
        """
        if depth == 0:
            return self.heuristic(node.state)

        # Expand children if not already expanded
        children = node.compute_and_get_children()
        if len(children) == 0:
            return self.heuristic(node.state)

        if is_maximizing:
            max_eval = -float('inf')
            for child in children:
                eval = self.minimax(child, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    logging.debug("Alpha-Beta Pruning (Maximizing Player)")
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for child in children:
                eval = self.minimax(child, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    logging.debug("Alpha-Beta Pruning (Minimizing Player)")
                    break
            return min_eval

    def calculate_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance with wrapping on x-axis
        The game board wraps horizontally (20 cells wide)
        """
        dx = min((pos1[0] - pos2[0]) % 20, (pos2[0] - pos1[0]) % 20)  # Handle wrapped x-axis
        dy = abs(pos1[1] - pos2[1])  # Regular y-axis distance
        return dx + dy

    def heuristic(self, state: State) -> float:
        """
        Improved heuristic using better distance calculation
        """
        # Score weight increased for better balance
        score_diff = 3.0 * (state.get_player_scores()[0] - state.get_player_scores()[1])
        
        hook_pos = state.get_hook_positions()[0]
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        
        # Calculate weighted distance to fish based on their scores
        position_value = 0
        if fish_positions:
            for fish_id, fish_pos in fish_positions.items():
                distance = self.calculate_distance(hook_pos, fish_pos)
                fish_score = fish_scores[fish_id]
                position_value += fish_score / (distance + 1)
        
        caught = state.get_caught()
        caught_bonus = 0
        if caught[0] is not None:
            caught_bonus = 10
        if caught[1] is not None:
            caught_bonus = -10

        return score_diff + (position_value * 2) + caught_bonus
