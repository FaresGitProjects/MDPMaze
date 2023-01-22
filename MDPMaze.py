
import numpy as np
from markov import *
import random

# Actions of Maze problem
actions = ['up', 'left', 'down', 'right', 'stop']


class MDPMaze:
    def __init__(self, maze, stateReward):

        self.maze = maze
        self.stateReward = stateReward
        self.stateSize = maze.stateSize
        self.stateReward.resize(self.stateSize)

        self.eps = 0.30
        self.gamma = 0.9
        self.rewardM = np.ones(self.stateSize) * (-1)

        # place holders for computing transition matrices
        self.Aup = None
        self.Aleft = None
        self.Adown = None
        self.Aright = None
        self.Astop = None

        # computeTransitionMatrices function should compute self.Aup, self.Aleft, self.Adown, self.Aright and self.Astop
        # update the 5 matrices inside computeTransitionMatrices()
        self.computeTransitionMatrices()

        self.value = np.zeros(self.stateSize)
        self.policy = []

        # You can use this to construct the noisy matrices

    def ARandomWalk(self):
        A = np.zeros((self.stateSize, self.stateSize))

        for col in range(self.stateSize):
            nbrs = self.maze.nbrList(col)
            p = 1 / (len(nbrs) + 1)
            A[col, col] = p
            for r in nbrs:
                A[r, col] = p
        return A

    def computeTransitionMatrices(self):
        Arandom = self.ARandomWalk()

        Aup_perfect = np.zeros((self.stateSize, self.stateSize))
        Aleft_perfect = np.zeros((self.stateSize, self.stateSize))
        Adown_perfect = np.zeros((self.stateSize, self.stateSize))
        Aright_perfect = np.zeros((self.stateSize, self.stateSize))
        Astop_perfect = np.zeros((self.stateSize, self.stateSize))

        for i in range(0, self.stateSize):
            action = self.maze.actionList(i)
            position = self.maze.nbrList(i)

            # Aup_perfect
            if 'U' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Aup_perfect[cord][i] = 1
            else:
                p = action.index('U')
                Aup_perfect[position[p]][i] = 1

            # Aleft_perfect
            if 'L' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Aleft_perfect[cord][i] = 1
            else:
                p = action.index('L')
                Aleft_perfect[position[p]][i] = 1

            # Adown_perfect
            if 'D' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Adown_perfect[cord][i] = 1
            else:
                p = action.index('D')
                Adown_perfect[position[p]][i] = 1

            # Aright_perfect
            if 'R' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Aright_perfect[cord][i] = 1
            else:
                p = action.index('R')
                Aright_perfect[position[p]][i] = 1

            # Astop_perfect
            r, c = self.maze.state2coord(i)
            cord = self.maze.coord2state((r, c))
            Astop_perfect[cord][i] = 1

        self.Aup = ((1 - self.eps) * Aup_perfect) + (self.eps * Arandom)
        self.Aleft = ((1 - self.eps) * Aleft_perfect) + (self.eps * Arandom)
        self.Adown = ((1 - self.eps) * Adown_perfect) + (self.eps * Arandom)
        self.Aright = ((1 - self.eps) * Aright_perfect) + (self.eps * Arandom)
        self.Astop = Astop_perfect

    def valIter(self):
        ''' YOUR CODE HERE
        Update self.value
        '''
        v_prime = np.zeros(self.stateSize)
        # R, T, gamma, all established
        while True:
            v = v_prime.copy()
            delta = 0
            for s in range(0, self.stateSize):
                u_Up = self.stateReward[s] + self.gamma * np.dot(self.Aup[:, s], self.value) - 1
                u_Left = self.stateReward[s] + self.gamma * np.dot(self.Aleft[:, s], self.value) - 1
                u_Down = self.stateReward[s] + self.gamma * np.dot(self.Adown[:, s], self.value) - 1
                u_Right = self.stateReward[s] + self.gamma * np.dot(self.Aright[:, s], self.value) - 1
                u_Stop = self.stateReward[s] + self.gamma * np.dot(self.Astop[:, s], self.value)

                v_prime[s] = max(u_Up, u_Down, u_Left, u_Right, u_Stop)
                delta = max(delta, abs(v_prime[s] - v[s]))

            if delta < self.eps * (1 - self.gamma) / self.gamma:
                self.value = v
                return self.value

    def polIter(self):
        ''' YOUR CODE HERE
        Update self.policy
        '''
        pi_prime = [random.choice(self.maze.actionList(s)) for s in range(self.stateSize)]
        while True:
            pi = pi_prime.copy()
            unchanged = True
            for s in range(self.stateSize):
                u_Up = self.stateReward[s] + self.gamma * np.dot(self.Aup[:, s], self.value) - 1
                u_Left = self.stateReward[s] + self.gamma * np.dot(self.Aleft[:, s], self.value) - 1
                u_Down = self.stateReward[s] + self.gamma * np.dot(self.Adown[:, s], self.value) - 1
                u_Right = self.stateReward[s] + self.gamma * np.dot(self.Aright[:, s], self.value) - 1
                u_Stop = self.stateReward[s] + self.gamma * np.dot(self.Astop[:, s], self.value)

                action = max(u_Up, u_Down, u_Left, u_Right, u_Stop)

                if action == u_Up:
                    action = 'up'
                if action == u_Down:
                    action = 'down'
                if action == u_Right:
                    action = 'right'
                if action == u_Left:
                    action = 'left'
                if action == u_Stop:
                    action = 'stop'

                if pi_prime[s] != action:
                    pi_prime[s] = action
                    unchanged = False

            if unchanged:
                self.policy = pi
                return self.policy


# ------------------------------------------------------------- #
if __name__ == "__main__":
    myMaze = maze(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]))

    stateReward = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
        [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]])

    mdp = MDPMaze(myMaze, stateReward)

    iterCount = 100
    printSkip = 10
    for i in range(iterCount):
        mdp.valIter()
        mdp.polIter()
        if np.mod(i, printSkip) == 0:
            print("Iteration ", i)
            print(mdp.policy)
            print(mdp.value)
