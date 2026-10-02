"""
Tic-Tac-Toe Q-Learning Agent
============================
A tabular Q-learning agent that learns to play Tic-Tac-Toe through self-play.

Usage:
    python tictactoe_qlearning.py train      # train the agent and save qtable.json
    python tictactoe_qlearning.py play       # play against the trained agent
"""

import random
import json
import sys

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


# ---------------------------------------------------------------------------
# Game environment
# ---------------------------------------------------------------------------

class TicTacToe:
    """A single game of Tic-Tac-Toe. State is a 9-cell tuple of 'X'/'O'/' '."""

    def __init__(self):
        self.board = [' '] * 9

    def get_state(self):
        return tuple(self.board)

    def current_player(self):
        x_count = self.board.count('X')
        o_count = self.board.count('O')
        return 'X' if x_count == o_count else 'O'   # X always moves first

    @staticmethod
    def legal_actions(state):
        return [i for i, cell in enumerate(state) if cell == ' ']

    def make_move(self, action):
        if self.board[action] != ' ':
            raise ValueError(f"Cell {action} is already taken")
        self.board[action] = self.current_player()

    def winner(self):
        for a, b, c in WIN_LINES:
            if self.board[a] != ' ' and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_draw(self):
        return ' ' not in self.board and self.winner() is None

    def is_over(self):
        return self.winner() is not None or self.is_draw()

    def render(self):
        def cell(i):
            return self.board[i] if self.board[i] != ' ' else str(i)
        rows = [' | '.join(cell(r * 3 + c) for c in range(3)) for r in range(3)]
        print(('\n' + '-' * 9 + '\n').join(rows))


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999):
        self.Q = {}                    # (state, action) -> value
        self.alpha = alpha             # learning rate
        self.gamma = gamma             # discount factor
        self.epsilon = epsilon         # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, legal_actions, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal_actions)          # explore
        q_values = [self.get_Q(state, a) for a in legal_actions]
        max_q = max(q_values)
        best = [a for a, q in zip(legal_actions, q_values) if q == max_q]
        return random.choice(best)                        # exploit, break ties randomly

    def update(self, state, action, reward, next_state, next_legal_actions):
        if next_state is None or not next_legal_actions:
            max_next_q = 0.0                               # terminal state
        else:
            max_next_q = max(self.get_Q(next_state, a) for a in next_legal_actions)
        old_q = self.get_Q(state, action)
        target = reward + self.gamma * max_next_q
        self.Q[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        table = {}
        for (state, action), value in self.Q.items():
            table.setdefault(''.join(state), {})[str(action)] = value
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(table, f)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            table = json.load(f)
        self.Q = {
            (tuple(state), int(action)): value
            for state, actions in table.items()
            for action, value in actions.items()
        }


# ---------------------------------------------------------------------------
# Training (self-play)
# ---------------------------------------------------------------------------

def play_training_episode(agent):
    """Play one self-play game; return each player's (state, action) sequence
    and the winner ('X', 'O', or None for a draw)."""
    game = TicTacToe()
    moves = {'X': [], 'O': []}
    while not game.is_over():
        player = game.current_player()
        state = game.get_state()
        legal = game.legal_actions(state)
        action = agent.choose_action(state, legal)
        game.make_move(action)
        moves[player].append({'state': state, 'action': action})
    return moves, game.winner()


def assign_updates(agent, moves, player, winner):
    """Bellman-update one player's move sequence. Reward is 0 until the final
    move; earlier moves bootstrap off that player's *next* own decision point."""
    seq = moves[player]
    for i, step in enumerate(seq):
        state, action = step['state'], step['action']
        if i + 1 < len(seq):
            next_state = seq[i + 1]['state']
            next_legal = TicTacToe.legal_actions(next_state)
            reward = 0
        else:
            next_state, next_legal = None, []
            if winner is None:
                reward = 0
            elif winner == player:
                reward = 1
            else:
                reward = -1
        agent.update(state, action, reward, next_state, next_legal)


def evaluate(agent, num_games=500):
    """Greedy agent vs. a random opponent, alternating who moves first."""
    wins = losses = draws = 0
    for i in range(num_games):
        game = TicTacToe()
        agent_mark = 'X' if i % 2 == 0 else 'O'
        while not game.is_over():
            state = game.get_state()
            legal = game.legal_actions(state)
            if game.current_player() == agent_mark:
                action = agent.choose_action(state, legal, greedy=True)
            else:
                action = random.choice(legal)
            game.make_move(action)
        w = game.winner()
        if w == agent_mark:
            wins += 1
        elif w is None:
            draws += 1
        else:
            losses += 1
    return wins / num_games, losses / num_games, draws / num_games


def train(agent, episodes=100_000, eval_every=10_000, eval_games=500):
    for ep in range(1, episodes + 1):
        moves, winner = play_training_episode(agent)
        assign_updates(agent, moves, 'X', winner)
        assign_updates(agent, moves, 'O', winner)
        agent.decay_epsilon()

        if ep % eval_every == 0:
            win, loss, draw = evaluate(agent, eval_games)
            print(f"Episode {ep:>7,} | eps={agent.epsilon:.3f} | "
                  f"vs random -> win {win:6.1%}  loss {loss:6.1%}  draw {draw:6.1%}")
    return agent


# ---------------------------------------------------------------------------
# Play against a human
# ---------------------------------------------------------------------------

def play_against_human(agent, q_table_path='qtable.json'):
    agent.load(q_table_path)
    game = TicTacToe()
    human_mark = input("Play as X or O? (X moves first): ").strip().upper()
    if human_mark not in ('X', 'O'):
        human_mark = 'X'
    print(f"You are {human_mark}. Cells are numbered 0-8.\n")

    while not game.is_over():
        game.render()
        state = game.get_state()
        legal = game.legal_actions(state)
        if game.current_player() == human_mark:
            action = None
            while action not in legal:
                try:
                    action = int(input("Your move (0-8): "))
                except ValueError:
                    continue
        else:
            action = agent.choose_action(state, legal, greedy=True)
            print(f"Agent plays {action}")
        game.make_move(action)
        print()

    game.render()
    w = game.winner()
    if w == human_mark:
        print("You win!")
    elif w is None:
        print("Draw.")
    else:
        print("Agent wins!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'play'):
        print("Usage: python tictactoe_qlearning.py [train|play]")
        return

    q_path = 'qtable.json'
    if sys.argv[1] == 'train':
        agent = QLearningAgent()
        train(agent, episodes=100_000, eval_every=10_000, eval_games=500)
        agent.save(q_path)
        print(f"\nSaved Q-table ({len(agent.Q)} entries) to {q_path}")
    else:
        agent = QLearningAgent()
        play_against_human(agent, q_path)


if __name__ == '__main__':
    main()
