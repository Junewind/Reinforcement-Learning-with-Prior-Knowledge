# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:33:56 2019

@author: Jisoo
"""

import numpy as np
import sys
from gym.envs.toy_text import discrete
from contextlib import closing
from six import StringIO, b
from gym import utils

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "standard": [
        "WWWWWWWWWWWWWW",
        "WPWPPPWPPPPPGW",
        "WPWPWPWWPWWWWW",
        "WPWPWPPWPWWWWW",
        "WPWPWWPWPPPPPW",
        "WPPPPWPWWWWWPW",
        "WWWWPWPPPPPWPW",
        "WWPWPWWWPWWWPW",
        "WWPWPPPWPPPPPW",
        "WWPWPWPWPWWWWW",
        "WPPPPWPWPPPPPW",
        "WWWWPWPWWWWWPW",
        "WSPPPWPPPPPPPW",
        "WWWWWWWWWWWWWW"
    ],
    "standard2": [
        "WWWWWWWWWWW",
        "WPWPPPPPPGW",
        "WPWPWPWWWWW",
        "WPPPWPPPPPW",
        "WPWWWWWWWPW",
        "WPPPPPPWPPW",
        "WPWWWWPWPWW",
        "WPPPPWPPPPW",
        "WWWWPWPWWWW",
        "WSPPPPPPPPW",
        "WWWWWWWWWWW",
    ],
}

class MazeEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc = None, map_name="standard2"):
        desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-100, 1000)
        
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
                    else:
                        current_row = row
                        current_col = col
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = bytes(newletter) in b'G'
                        if (newletter in b'G'):
                            rew = 30
                        elif (newletter in b'W'):
                            newstate = to_s(current_row, current_col)
                            newletter = desc[current_row, current_col]
                            rew = -20
                        else:
                            rew = -1
                        li.append((1.0, newstate, rew, done))

        super(MazeEnv, self).__init__(nS, nA, P, isd)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()