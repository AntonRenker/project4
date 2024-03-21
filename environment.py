import numpy as np

class Environment:
    def __init__(self, num_columns, num_rows, num_win) -> None:
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_win = num_win
        self.board = np.zeros(num_columns * num_rows)

    def reset(self) -> tuple:
        self.board = np.zeros(self.num_columns * self.num_rows)
        return self.board

    def __check_action(self, action) -> bool:
        return self.board[self.num_rows * self.num_columns - (self.num_columns - action)] == 0

    def __update_board(self, action, player) -> None:
        for i in range(action, self.num_columns * self.num_rows, self.num_columns):
            if self.board[i] == 0:
                self.board[i] = player
                break

    def __check_win(self, player) -> bool:
        # Check rows
        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j * self.num_columns < self.num_columns * self.num_rows and self.board[i + j * self.num_columns] == player:
                        count += 1
                if count == self.num_win:
                    return True

        # Check columns
        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j < self.num_columns * self.num_rows and self.board[i + j] == player:
                        count += 1
                if count == self.num_win:
                    return True

        # Check diagonals
        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j * (self.num_columns + 1) < self.num_columns * self.num_rows and self.board[i + j * (self.num_columns + 1)] == player:
                        count += 1
                if count == self.num_win:
                    return True

        for i in range(self.num_columns * self.num_rows):
            if self.board[i] == player:
                count = 1
                for j in range(1, self.num_win):
                    if i + j * (self.num_columns - 1) < self.num_columns * self.num_rows and self.board[i + j * (self.num_columns - 1)] == player:
                        count += 1
                if count == self.num_win:
                    return True

        return False
                
    def __get_action(self, Q, epsilon) -> int:
        possible_actions = [i for i in range(self.num_columns) if self.__check_action(i)]
        if np.random.rand() < epsilon:
            action = np.random.choice(possible_actions)
        else:
            state = np.array([self.board])
            act_values = Q.predict(state)
            action = np.argmax(act_values[0])
            if action not in possible_actions:
                action = np.random.choice(possible_actions)
        return action

    def step(self, action, player, Q, epsilon) -> tuple:
        # Check if the action is valid
        if self.__check_action(action):
            # Update the board
            self.__update_board(action, player)
            # Check if the game is over
            if self.__check_win(player):
                return self.board, 1, True
            elif np.all(self.board != 0):
                return self.board, 0, True
            else:
                # Do epsilon greedy move based on Q
                action = self.__get_action(Q, epsilon)
                self.__update_board(action, -player)
                # Check if the game is over
                if self.__check_win(-player):
                    return self.board, -1, True
                elif np.all(self.board != 0):
                    return self.board, 0, True
                else:
                    # Game is not over
                    return self.board, 0, False
        else:
            return self.board, -100, True
        
    def single_step(self, action, player) -> tuple:
        # Check if the action is valid
        if self.__check_action(action):
            # Update the board
            self.__update_board(action, player)
            # Check if the game is over
            if self.__check_win(player):
                return self.board, 1, True
            elif np.all(self.board != 0):
                return self.board, 0, True
            else:
                # Game is not over
                return self.board, 0, False
        else:
            return self.board, -100, True
        
    def render(self) -> None:
        for row in range(self.num_rows - 1, -1, -1):
            print("|", end="")
            for col in range(self.num_columns):
                if self.board[row * self.num_columns + col] == 1:
                    print(" X ", end="|")
                elif self.board[row * self.num_columns + col] == -1:
                    print(" O ", end="|")
                else:
                    print("   ", end="|")
            print()
        print("-" * (self.num_columns * 4 + 1))
        # print(" " + " ".join([str(i) for i in range(self.num_columns)]))

