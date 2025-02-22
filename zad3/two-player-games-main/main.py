from two_player_games.player import Player
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove
from MinMaxSolver import MinMaxSolver
import matplotlib.pyplot as plt
import time


def game_simulation(column_count: int, row_count: int, p1: Player, p2: Player, depth: int, heuristic: int, print_flag: bool):
    alpha = float("-inf")
    beta = float("inf")
    is_maximizing_player = True
    is_game_finished_flag = False
    game = ConnectFour(size=(column_count, row_count), first_player=p1, second_player=p2)
    algorithm = MinMaxSolver(game, heuristic)
    while (not is_game_finished_flag):
        if game.is_finished():
            is_game_finished_flag = True
            continue
        best_move_column, _ = algorithm.minimax(depth, alpha, beta, is_maximizing_player)
        game.make_move(ConnectFourMove(best_move_column))
        is_maximizing_player = not is_maximizing_player
        if print_flag:
            time.sleep(2)
            print(game)
    if print_flag:
        print("End of the game.")
        try:
            print(f"Winner: {game.get_winner().char}")
        except Exception:
            print("Draw")


def count_time(column_count: int, row_count: int, p1: Player, p2: Player, depth: int, heuristic: int, print_flag: bool = False, iterations: int = 10):
    average_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        game_simulation(column_count, row_count, p1, p2, depth, heuristic, print_flag)
        end_time = time.perf_counter()
        average_time += end_time - start_time
    average_time /= iterations
    return average_time


def create_plot(column_count: int, row_count: int, p1: Player, p2: Player, heuristic: int):
    PRINT_FLAG = False
    ITERATIONS = 5
    depths = [1, 2, 3, 4, 5]
    times = []
    for depth in depths:
        times.append(count_time(column_count, row_count, p1, p2, depth, heuristic, PRINT_FLAG, ITERATIONS))
    plt.plot(depths, times)
    plt.title(f"Minimax a - b algorithm in 'Connect Four' game, heuristic = {heuristic} ")
    plt.xlabel("Depth")
    plt.ylabel("Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_heuristic_{heuristic}.png")
    plt.show()


def main():
    ROW_COUNT = 6
    COLUMN_COUNT = 7
    p1 = Player("a")
    p2 = Player("b")
    heuristic = 3
    depth = 6
    print_flag = False
    create_plot(COLUMN_COUNT, ROW_COUNT, p1, p2, heuristic)


if __name__ == "__main__":
    main()
