# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:35:39 2019

@author: Jisoo
"""

import numpy as np
import sys
from gym.envs.toy_text import discrete
from contextlib import closing
from six import StringIO, b
from gym import utils

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {
    "standard": [
        "WWWWWWWWWWWW",
        "WPPPPPPPPPPW",
        "WPPPPPPPPPPW",
        "WPPPPPPPPPPW",
        "WSCCCCCCCCGW",
        "WWWWWWWWWWWW"
    ],
}

class CliffWalkingEnv(discrete.DiscreteEnv):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.
    Adapted from Example 6.6 (page 106) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf
    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py
    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start. An episode terminates when the agent reaches the goal.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc = None, map_name="standard"):
        desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-100, 100)
        
        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()
        
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        
        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'G':
                        li.append((1.0, s, 100, True))
                    elif letter in b'C':
                        li.append((1.0, s, -100, True))
#                    elif letter in b'W':
#                        li.append((1.0, s, -10, True))
                    else:
                        current_row = row
                        current_col = col
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = bytes(newletter) in b'GC'
                        if (newletter in b'G'):
                            rew = 100
                        elif (newletter in b'C'):
                            rew = -100
                        elif (newletter in b'W'):
                            newstate = to_s(current_row, current_col)
                            newletter = desc[current_row, current_col]
                            rew = -10
                        else:
                            rew = -1
                        li.append((1.0, newstate, rew, done))

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Up","Right","Down","Left"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()