import random
import copy
import math
from collections import namedtuple

class Checkers():
    X = "X"
    O = "O"

class board:
    def __init__(self, board_size=None, turn_x=True, print_board=False):
        if board_size is None:
            self.board_size = namedtuple('board_size', 'width height')(7,6)
        else:
            self.board_size = board_size
        self.turn_x = turn_x
        self.board = [list() for _ in range(self.board_size.width)]
        self.print_board = print_board
        self.completed = False
        self.drawn = False

    def __getitem__(self, pos):
        if not (pos[0] >= 0 and pos[1] >= 0):
            return False
        try:
            return self.board[pos[0]][pos[1]]
        except IndexError:
            return False

    def print(self, message=''):
        if self.print_board:
            self.force_print(message)

    def force_print(self, message=''):
        print(message)
        print(
            '|{}|'.format('|'.join([str(x) for x in range(self.board_size.width)])))
        for i in range(self.board_size.height)[::-1]:
            print('|{}|'.format(
                '|'.join([col[i] if i < len(col) else ' ' for col in self.board])))
        print("-" * (2 * self.board_size.width + 1))

    def get_piece(self):
        if self.turn_x:
            return Checkers.X
        else:
            return Checkers.O

    def check_win_col(self, col, win_len=4):
        pos = (col, self.col_last_index(col))
        return self.check_win_pos(pos, win_len)

    def play(self, col):
        assert self.board_size.height > len(
            self.board[col]), "Error -- move played on full slot"
        self.board[col].append(self.get_piece())
        self.turn_x = not self.turn_x
        self.completed = self.check_win_col(col)
        if self.valid_moves() == list() and not self.completed:
            self.completed = True
            self.drawn = True
        return self.completed

    def valid_moves(self):
        return [col
                for col in range(self.board_size.width)
                if self.board_size.height > len(self.board[col])]

    def col_last_index(self, col):
        return len(self.board[col]) - 1

    def check_win_pos(self, pos, win_len=4):
        def check_win_orient(pos, orient, win_len=4):
            checker = self[pos]
            counter = 1
            curr_pos = pos
            while True:
                curr_pos = tuple(sum(arr) for arr in zip(orient, curr_pos))
                if self[curr_pos] == checker:
                    counter += 1
                else:
                    break
            curr_pos = pos
            back_orient = tuple(-x for x in orient)
            while True:
                curr_pos = tuple(sum(arr) for arr in zip(back_orient, curr_pos))
                if self[curr_pos] == checker:
                    counter += 1
                else:
                    break
            return counter >= win_len

        check_directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for step in check_directions:
            if check_win_orient(pos, step, win_len):
                return True
        return False

    def human_turn(self):
        move = None
        while move==None:
            try:
                move = int(input("play a move:"))
                if move not in range(self.board_size.width):
                    raise ValueError
                self.play(move)
            except ValueError:
                print("Please play a valid move.")
                move = None

    def random_play_static(self):
        board_bkup, turn_bkup, completed_bkup = \
            (copy.deepcopy(self.board), copy.deepcopy(self.turn_x), self.completed)
        output = self.random_play()
        self.print()
        self.board, self.turn_x, self.completed = \
            (board_bkup, turn_bkup, completed_bkup)
        if self.turn_x:
            return output
        else:
            return 1 - output

    def random_play(self):
        if self.completed:
            return (0.5 if self.drawn else (0 if self.turn_x else 1))
        finished = False
        while not finished:
            valid_moves = self.valid_moves()
            if len(valid_moves) == 0:
                return 0.5
            move_ind = random.randint(0, len(valid_moves) - 1)
            finished = self.play(valid_moves[move_ind])
        if self.turn_x:
            return 0
        else:
            return 1

    def eval_fitness(self, f = None):
        if f == None:
            return self.random_play_static()
        else:
            if self.turn_x:
                return foutput
            else:
                return 1 - output



class tree_node:
    def __init__(self, board, parent=None, tree_search=None, move=None):
        self.move = move
        self.board = board
        self.children = []
        self.parent = parent
        self.total_count = 0
        self.win_count = 0.0
        self.valid_moves = set(board.valid_moves())
        self.tree_search = tree_search

        if tree_search:
            tree_search.nodes.append(self)

    def back_prop(self, total_inc, inc):
        assert total_inc >= inc
        self.total_count += total_inc
        self.win_count += inc
        if self.parent:
            self.parent.back_prop(total_inc, total_inc - inc)

    def play(self):
        self.back_prop(1, self.board.random_play_static())

    def create_node(self, move):
        if self.board.completed:
            self.play()
        else:
            assert move in self.valid_moves, "Error move not valid in MCTS"

            self.valid_moves.remove(move)

            new_board = copy.deepcopy(self.board)
            new_board.play(move)

            new_node = tree_node(new_board, self, self.tree_search, move)
            self.children.append(new_node)
            new_node.play()

    def random_expansion(self):
        move = random.sample(self.valid_moves, 1)[0]
        self.create_node(move)


class MCTS:
    def __init__(self, board, max_turns):
        self.max_turns = max_turns
        self.nodes = []
        self.root = tree_node(board, parent=None, tree_search=self)
        self.board = board

    def expansion_random(self, turn):
        def sel(node):
            if len(node.valid_moves) == 0:
                choice_vals = [self.choice_value(child, turn) for child in
                               node.children]
                if len(choice_vals) == 0:
                    node.play()
                else:
                    max_ind = choice_vals.index(max(choice_vals))
                    sel(node.children[max_ind])
            else:
                node.random_expansion()

        sel(self.root)

    def choose_move(self):
        for i in range(self.max_turns):
            self.expansion_random(i + 1)
        move_nodes = [1 - (node.win_count / node.total_count) for node in
                      self.root.children]
        max_ind = move_nodes.index(max(move_nodes))
        if self.board.turn_x:
            print('X win probability:{}'.format(max(move_nodes)))
            print('O Win probability:{}'.format(1 - max(move_nodes)))
        else:
            print('X win probability:{}'.format(1 - max(move_nodes)))
            print('O Win probability:{}'.format(max(move_nodes)))

        return self.root.children[max_ind].move

    def choice_value(self, node, turn, c=None):

        if not c:
            c = math.sqrt(2)
        n = node.total_count
        w = node.win_count
        N = turn
        return c * math.sqrt(math.log(N) / n) + 1 - (w / n)

    def add_node(self, node):
        self.nodes.append(node)

def population():
    def duel(self,ai1, ai2):

        return True
if __name__ == "__main__":
    x_win=0
    for i in range (100):
        game = board()
        game.force_print()
        while not game.completed:
            if game.turn_x:
                game.human_turn()
                game.force_print()
            else:
                max_turns = 2000
                mcts = MCTS(copy.deepcopy(game), max_turns)
                game.play(mcts.choose_move())
                game.force_print()
        if game.drawn:
            print("Everyone loses")
            x_win += 0.5
        else:
            if game.turn_x:
                print("O wins")
                x_win+=1
            else:
                print("X wins")
        print('Total games: {}'.format(i))
        print('Points to X: {}'.format(x_win))
