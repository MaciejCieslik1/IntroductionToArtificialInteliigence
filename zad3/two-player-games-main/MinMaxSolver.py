from typing import Tuple, List
from two_player_games.player import Player
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove


class MinMaxSolver:

    def __init__(self, game: ConnectFour, heuristic: int):
        self.game = game
        self.heuristic = heuristic

    def count_value(self, is_maximizing_player: bool) -> float:
        if self.heuristic == 1:
            return self.count_value1(is_maximizing_player)
        elif self.heuristic == 2:
            return self.count_value2(is_maximizing_player)
        else:
            return self.count_value3(is_maximizing_player)

    def count_value1(self, is_maximizing_player: bool) -> float:
        FIELDS_VALUES = [2, 3, 4, 3, 2, 1, 4, 5, 7, 6, 4, 2, 5, 8, 9, 8, 6, 3, 6, 9, 10, 10, 7, 5, 5, 8, 9, 8, 6, 3, 4, 5, 7, 6, 4, 2, 2, 3, 4, 3, 2, 1]
        field_chars_table = self.create_field_chars_table()
        points = 0
        multiplier = 1
        player = self.game.first_player
        if not is_maximizing_player:
            multiplier = -1
            player = self.game.second_player
        for index, field_char in enumerate(field_chars_table):
            if field_char == player.char:
                points += FIELDS_VALUES[index]
        return multiplier * points

    def count_value2(self, is_maximizing_player: bool) -> float:
        points = 0
        multiplier = 1
        player = self.game.first_player
        if not is_maximizing_player:
            multiplier = -1
            player = self.game.second_player
        points = self.game.state.get_score(player)
        return multiplier * points

    def count_value3(self, is_maximizing_player: bool) -> float:
        points = 0
        multiplier = 1
        player = self.game.first_player
        if not is_maximizing_player:
            multiplier = -1
            player = self.game.second_player
        points = self.game.state.get_score(player)
        return multiplier * points * points

    def is_valid_move(self, col_index: int) -> bool:
        fields = self.create_field_chars_table()
        FIELDS_IN_COLUMN = 6
        OFFSET = 5
        if fields[col_index * FIELDS_IN_COLUMN + OFFSET] == 0:
            return True
        else:
            return False

    def create_field_chars_table(self) -> List[str]:
        """Return list of characters on each field. Each letter position is calculated, using its index"""
        fields = [field.char if field is not None else "o" for column in self.game.state.fields for field in column]
        return fields

    def minimax(self, depth, alpha: float, beta: float, is_maximizing_player: bool) -> Tuple[int, float]:
        """Returns column index and score"""
        if depth == 0 or self.game.is_finished():
            return None, self.count_value(is_maximizing_player)
        best_move = None
        if is_maximizing_player:
            max_score = float("-inf")
            for move in self.game.get_moves():
                self.game.make_move(move)
                _, current_score = self.minimax(depth - 1, alpha, beta, not is_maximizing_player)
                self.game.undo_move(move)
                if current_score > max_score:
                    max_score = current_score
                    best_move = move
                alpha = max(alpha, current_score)
                if alpha >= beta:
                    break

            return best_move.column, max_score
        else:
            min_score = float("inf")
            for move in self.game.get_moves():
                self.game.make_move(move)
                _, current_score = self.minimax(depth - 1, alpha, beta, not is_maximizing_player)
                self.game.undo_move(move)
                if current_score < min_score:
                    min_score = current_score
                    best_move = move
                beta = min(beta, current_score)
                if alpha >= beta:
                    break
            return best_move.column, min_score
