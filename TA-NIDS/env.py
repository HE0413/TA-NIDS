# Wrapper of environment
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines import bench, logger

import numpy as np
import heapq
import random

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from TA-NIDS.utils import read_data_as_matrix, run_iforest
from stable_baselines.common import make_vec_env


class EnvBase(gym.Env):
    def __init__(self, datapath="data/toy.csv", budget=100):
        # Read dataset
        self.wocnnn = 0
        X_train, labels, anomalies = read_data_as_matrix(datapath)
        self.X_train = X_train
        self.labels = labels
        self.size = len(self.labels)
        self.budget = budget
        self.dim = X_train.shape[1]
        self.state_dim = 30

        # Unsupervised scores
        #self.X_t = StandardScaler().fit_transform(self.X_train)
        self.scores = np.expand_dims(run_iforest(self.X_train), axis=1)

        woc1 = np.mean(self.scores, axis=0)[None, :]
        woc2 = np.std(self.scores, axis=0)[None, :]

        self.scores = (self.scores - np.mean(self.scores, axis=0)[None, :]) / np.std(self.scores, axis=0)[None, :]

        # Exatract distances features
        self.X_train = StandardScaler().fit_transform(self.X_train)
        self.distances = euclidean_distances(self.X_train, self.X_train)

        woc1 = np.mean(self.distances, axis=1)[:, None]
        woc2 = np.std(self.distances, axis=1)[:, None]

        self.distances = (self.distances - np.mean(self.distances, axis=1)[:, None]) / np.std(self.distances, axis=1)[:,
                                                                                       None]
        self.nearest_neighbors = np.argpartition(self.distances, 10)[:, :10]

        print("Data loaded: {} Total instances: {} Anomalies: {}".format(datapath, self.size, len(anomalies)))

        # Gym settings
        self.action_space = spaces.Discrete(10)
        high = np.ones(self.state_dim) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass


class TrainEnv(EnvBase):

    def __init__(self, datapath):
        super().__init__(datapath=datapath)

    def step(self, action):
        """ Proceed to the next state given the curernt action
            1 for check the instance, 0 for not
            return next state, reward and done
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if action == 9:
            if self._Y[self.pointer] == 9:
                r = 0
                self.normalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 0:
                    self.anomalies0.append(self.pointer)
                elif self._Y[self.pointer] == 1:
                    self.anomalies1.append(self.pointer)
                elif self._Y[self.pointer] == 2:
                    self.anomalies2.append(self.pointer)
                elif self._Y[self.pointer] == 3:
                    self.anomalies3.append(self.pointer)
                elif self._Y[self.pointer] == 4:
                    self.anomalies4.append(self.pointer)
                elif self._Y[self.pointer] == 5:
                    self.anomalies5.append(self.pointer)
                elif self._Y[self.pointer] == 6:
                    self.anomalies6.append(self.pointer)
                elif self._Y[self.pointer] == 7:
                    self.anomalies7.append(self.pointer)
                elif self._Y[self.pointer] == 8:
                    self.anomalies8.append(self.pointer)
                self.anomalies.append(self.pointer)
        elif action == 0:
            if self._Y[self.pointer] == 0:
                r = 1
                self.anomalies0.append(self.pointer)
                self.anomalies.append(self.pointer)
            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 1:
            if self._Y[self.pointer] == 1:
                r = 1
                self.anomalies1.append(self.pointer)
                self.anomalies.append(self.pointer)
            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 2:
            if self._Y[self.pointer] == 2:
                r = 1
                self.anomalies2.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 3:
            if self._Y[self.pointer] == 3:
                r = 1
                self.anomalies3.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 4:
            if self._Y[self.pointer] == 4:
                r = 1
                self.anomalies4.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 5:
            if self._Y[self.pointer] == 5:
                r = 1
                self.anomalies5.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 6:
            if self._Y[self.pointer] == 6:
                r = 1
                self.anomalies6.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 7:
            if self._Y[self.pointer] == 7:
                r = 1
                self.anomalies7.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        elif action == 8:
            if self._Y[self.pointer] == 8:
                r = 1
                self.anomalies8.append(self.pointer)
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                if self._Y[self.pointer] == 9:
                    self.normalies.append(self.pointer)
                else:
                    if self._Y[self.pointer] == 0:
                        self.anomalies0.append(self.pointer)
                    elif self._Y[self.pointer] == 1:
                        self.anomalies1.append(self.pointer)
                    elif self._Y[self.pointer] == 2:
                        self.anomalies2.append(self.pointer)
                    elif self._Y[self.pointer] == 3:
                        self.anomalies3.append(self.pointer)
                    elif self._Y[self.pointer] == 4:
                        self.anomalies4.append(self.pointer)
                    elif self._Y[self.pointer] == 5:
                        self.anomalies5.append(self.pointer)
                    elif self._Y[self.pointer] == 6:
                        self.anomalies6.append(self.pointer)
                    elif self._Y[self.pointer] == 7:
                        self.anomalies7.append(self.pointer)
                    elif self._Y[self.pointer] == 8:
                        self.anomalies8.append(self.pointer)
                    self.anomalies.append(self.pointer)
            self.count += 1
        self.pointer += 1

        # Set maximum lenths to 2000
        if self.pointer >= self.size or self.pointer >= 2000:
            self.done = True
            #self.anomalies = []
            #self.anomalies0 = []
            #self.anomalies1 = []
            #self.anomalies2 = []
            #self.anomalies3 = []
            #self.anomalies4 = []
            #self.anomalies5 = []
            #self.anomalies6 = []
            #self.anomalies7 = []
            #self.anomalies8 = []
            #self.normalies = []

            a = self.indices[self.anomalies]
            a0 = self.indices[self.anomalies0]
            a1 = self.indices[self.anomalies1]
            a2 = self.indices[self.anomalies2]
            a3 = self.indices[self.anomalies3]
            a4 = self.indices[self.anomalies4]
            a5 = self.indices[self.anomalies5]
            a6 = self.indices[self.anomalies6]
            a7 = self.indices[self.anomalies7]
            a8 = self.indices[self.anomalies8]
            n = self.indices[self.normalies]
            import csv
            csv_file = open(r"/home/dell/new/wp/myown/mul3/TA-NIDS/TA-NIDS/loss/statenbs.csv", 'w')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(a)
            csv_writer.writerow(a0)
            csv_writer.writerow(a1)
            csv_writer.writerow(a2)
            csv_writer.writerow(a3)
            csv_writer.writerow(a4)
            csv_writer.writerow(a5)
            csv_writer.writerow(a6)
            csv_writer.writerow(a7)
            csv_writer.writerow(a8)
            csv_writer.writerow(n)
            csv_file.flush()
            print('woc')
            from pudb import set_trace
            set_trace()




        else:
            self.done = False

        return self._obs(), r, self.done, {}

    def reset(self):
        """ Reset the environment, for streaming evaluation
        """
        self._process_data()

        # Some stats
        self.pointer = 0
        self.count = 0
        self.done = False
        self.anomalies = []
        self.anomalies0 = []
        self.anomalies1 = []
        self.anomalies2 = []
        self.anomalies3 = []
        self.anomalies4 = []
        self.anomalies5 = []
        self.anomalies6 = []
        self.anomalies7 = []
        self.anomalies8 = []
        self.normalies = []
        self.labeled = []

        return self._obs()

    def _process_data(self):
        # Shuffle data
        self.indices = indices = np.random.choice(self.size, self.size, replace=False)
        self._X = wocx = self.X_train[indices]
        self._Y = wocy = self.labels[indices]
        self._scores = wocs = self.scores[indices]

    def _obs(self):
        """ Return the observation of the current state
        """
        if self.done:
            return np.zeros(self.state_dim)

        features = []
        ori_pointer = self.indices[self.pointer]
        ori_anomalies = self.indices[self.anomalies]
        ori_anomalies0 = self.indices[self.anomalies0]
        ori_anomalies1 = self.indices[self.anomalies1]
        ori_anomalies2 = self.indices[self.anomalies2]
        ori_anomalies3 = self.indices[self.anomalies3]
        ori_anomalies4 = self.indices[self.anomalies4]
        ori_anomalies5 = self.indices[self.anomalies5]
        ori_anomalies6 = self.indices[self.anomalies6]
        ori_anomalies7 = self.indices[self.anomalies7]
        ori_anomalies8 = self.indices[self.anomalies8]
        ori_nomalies = self.indices[self.normalies]
        
        '''''

        import csv
        with open(r"/home/dell/new/wp/myown/mul3/TA-NIDS/TA-NIDS/loss/statenb.csv", 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            wpnnnn = 0
            while wpnnnn < 11:
                rows[wpnnnn] = [int(x) for x in rows[wpnnnn]]
                wpnnnn = wpnnnn + 1
            ori_anomalies = rows[0]
            ori_anomalies0 = rows[1]
            ori_anomalies1 = rows[2]
            ori_anomalies2 = rows[3]
            ori_anomalies3 = rows[4]
            ori_anomalies4 = rows[5]
            ori_anomalies5 = rows[6]
            ori_anomalies6 = rows[7]
            ori_anomalies7 = rows[8]
            ori_anomalies8 = rows[9]
            ori_nomalies = rows[10]
        '''''

        near_anomalies = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies), 1, 0)
        near_anomalies0 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies0), 1, 0)
        near_anomalies1 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies1), 1, 0)
        near_anomalies2 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies2), 1, 0)
        near_anomalies3 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies3), 1, 0)
        near_anomalies4 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies4), 1, 0)
        near_anomalies5 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies5), 1, 0)
        near_anomalies6 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies6), 1, 0)
        near_anomalies7 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies7), 1, 0)
        near_anomalies8 = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies8), 1, 0)

        near_normalies = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_nomalies), 1, 0)
        #features.append(1) if np.count_nonzero(near_anomalies[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies0[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies1[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies2[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies3[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies4[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies5[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies6[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies7[:5]) > 0 else features.append(0)
        features.append(1) if np.count_nonzero(near_anomalies8[:5]) > 0 else features.append(0)

        # Avg distance to abnormal instances
        a = np.mean(self.distances[ori_pointer, ori_anomalies]) if len(self.anomalies) > 0 else 0
        a_min = np.min(self.distances[ori_pointer, ori_anomalies]) if len(self.anomalies) > 0 else 0
        a0 = np.mean(self.distances[ori_pointer, ori_anomalies0]) if len(self.anomalies0) > 0 and len(ori_anomalies0) > 0 else 0
        a0_min = np.min(self.distances[ori_pointer, ori_anomalies0]) if len(self.anomalies0) > 0 and len(ori_anomalies0) > 0 else 0
        a1 = np.mean(self.distances[ori_pointer, ori_anomalies1]) if len(self.anomalies1) > 0 and len(ori_anomalies1) > 0 else 0
        a1_min = np.min(self.distances[ori_pointer, ori_anomalies1]) if len(self.anomalies1) > 0 and len(ori_anomalies1) > 0 else 0
        a2 = np.mean(self.distances[ori_pointer, ori_anomalies2]) if len(self.anomalies2) > 0 and len(ori_anomalies2) > 0 else 0
        a2_min = np.min(self.distances[ori_pointer, ori_anomalies2]) if len(self.anomalies2) > 0 and len(ori_anomalies2) > 0 else 0
        a3 = np.mean(self.distances[ori_pointer, ori_anomalies3]) if len(self.anomalies3) > 0 and len(ori_anomalies3) > 0 else 0
        a3_min = np.min(self.distances[ori_pointer, ori_anomalies3]) if len(self.anomalies3) > 0 and len(ori_anomalies3) > 0 else 0
        a4 = np.mean(self.distances[ori_pointer, ori_anomalies4]) if len(self.anomalies4) > 0 and len(ori_anomalies4) > 0 else 0
        a4_min = np.min(self.distances[ori_pointer, ori_anomalies4]) if len(self.anomalies4) > 0 and len(ori_anomalies4) > 0 else 0
        a5 = np.mean(self.distances[ori_pointer, ori_anomalies5]) if len(self.anomalies5) > 0 and len(ori_anomalies5) > 0 else 0
        a5_min = np.min(self.distances[ori_pointer, ori_anomalies5]) if len(self.anomalies5) > 0 and len(ori_anomalies5) > 0 else 0
        a6 = np.mean(self.distances[ori_pointer, ori_anomalies6]) if len(self.anomalies6) > 0 and len(ori_anomalies6) > 0 else 0
        a6_min = np.min(self.distances[ori_pointer, ori_anomalies6]) if len(self.anomalies6) > 0 and len(ori_anomalies6) > 0 else 0
        a7 = np.mean(self.distances[ori_pointer, ori_anomalies7]) if len(self.anomalies7) > 0 and len(ori_anomalies7) > 0 else 0
        a7_min = np.min(self.distances[ori_pointer, ori_anomalies7]) if len(self.anomalies7) > 0 and len(ori_anomalies7) > 0 else 0
        a8 = np.mean(self.distances[ori_pointer, ori_anomalies8]) if len(self.anomalies8) > 0 and len(ori_anomalies8) > 0 else 0
        a8_min = np.min(self.distances[ori_pointer, ori_anomalies8]) if len(self.anomalies8) > 0 and len(ori_anomalies8) > 0 else 0

        # Avg distance to normal instances
        n = np.mean(self.distances[ori_pointer, ori_nomalies]) if len(self.normalies) > 0 else 0
        n_min = np.min(self.distances[ori_pointer, ori_nomalies]) if len(self.normalies) > 0 else 0

        features.extend(
            [a0, a0_min, a1, a1_min, a2, a2_min, a3, a3_min, a4, a4_min, a5, a5_min, a6, a6_min, a7, a7_min,
             a8, a8_min, n, n_min])
        c = self._scores[self.pointer]
        features.extend(c)

        return features


class EnsembleTrainEnv(gym.Env):
    def __init__(self, datapaths):
        self.envs = []
        self.nnn = 0
        for datapath in datapaths:
            self.envs.append(TrainEnv(datapath))

        # Gym settings
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.seed()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        for env in self.envs:
            env.seed(seed)
        return [seed]

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        #self.env = random.choice(self.envs)
        if self.nnn < 9:
            self.env = self.envs[self.nnn]
            self.nnn = self.nnn + 1
        else:
            #self.nnn = 0
            self.env = self.envs[self.nnn]
            self.nnn = 0
        return self.env.reset()

    def render(self):
        pass


class EvalEnv(EnvBase):

    def step(self, actionx, actiony):
        """ Evaluation step
        """
        assert actionx in self.legal
        if self._Y[actionx] != 9:
            if self._Y[actionx] != 9:#!= 9:#== actiony:
                r = 1
            else:
                r = 0
            if self._Y[actionx] == actiony:#!= 9:#== actiony:
                wocr = 1
                self.wocnnn = 0
            else:
                wocr = 0
                self.wocnnn = self.wocnnn + 1
            if self._Y[actionx] == 0:
                self.anomalies0.append(actionx)
            elif self._Y[actionx] == 1:
                self.anomalies1.append(actionx)
            elif self._Y[actionx] == 2:
                self.anomalies2.append(actionx)
            elif self._Y[actionx] == 3:
                self.anomalies3.append(actionx)
            elif self._Y[actionx] == 4:
                self.anomalies4.append(actionx)
            elif self._Y[actionx] == 5:
                self.anomalies5.append(actionx)
            elif self._Y[actionx] == 6:
                self.anomalies6.append(actionx)
            elif self._Y[actionx] == 7:
                self.anomalies7.append(actionx)
            elif self._Y[actionx] == 8:
                self.anomalies8.append(actionx)

            self.anomalies.append(actionx)
        else:
            if self._Y[actionx] == actiony:
                r = 0
                wocr = 0
            else:
                r = 0
                wocr = 0
            self.wocnnn = self.wocnnn + 1
            self.normalies.append(actionx)
        self.count += 1
        self.labeled.append(actionx)
        self.legal.remove(actionx)
        if self.count >= self.budget or self.wocnnn >= 100:
            self.done = True
        else:
            self.done = False
        s = self._obs()
        return s, self.legal, r, self.done, {}, wocr

    def reset(self):
        """ Evaluation reset
        """
        self._X = self.X_train
        self._Y = self.labels
        self._scores = self.scores

        # Some stats
        self.count = 0
        self.done = False
        self.anomalies = []
        self.anomalies0 = []
        self.anomalies1 = []
        self.anomalies2 = []
        self.anomalies3 = []
        self.anomalies4 = []
        self.anomalies5 = []
        self.anomalies6 = []
        self.anomalies7 = []
        self.anomalies8 = []
        self.normalies = []
        self.labeled = []
        self.legal = [i for i in range(self.size)]

        return self._obs(), self.legal

    def _obs(self):
        """ Return the observation of the current state
        """
        if self.done:
            return np.zeros(self.state_dim)

        near_anomalies = np.where(np.isin(self.nearest_neighbors, self.anomalies), 1, 0)
        near_anomalies0 = np.where(np.isin(self.nearest_neighbors, self.anomalies0), 1, 0)
        near_anomalies1 = np.where(np.isin(self.nearest_neighbors, self.anomalies1), 1, 0)
        near_anomalies2 = np.where(np.isin(self.nearest_neighbors, self.anomalies2), 1, 0)
        near_anomalies3 = np.where(np.isin(self.nearest_neighbors, self.anomalies3), 1, 0)
        near_anomalies4 = np.where(np.isin(self.nearest_neighbors, self.anomalies4), 1, 0)
        near_anomalies5 = np.where(np.isin(self.nearest_neighbors, self.anomalies5), 1, 0)
        near_anomalies6 = np.where(np.isin(self.nearest_neighbors, self.anomalies6), 1, 0)
        near_anomalies7 = np.where(np.isin(self.nearest_neighbors, self.anomalies7), 1, 0)
        near_anomalies8 = np.where(np.isin(self.nearest_neighbors, self.anomalies8), 1, 0)

        near_normalies = np.where(np.isin(self.nearest_neighbors, self.normalies), 1, 0)
        a_top_5 = np.where(np.count_nonzero(near_anomalies[:, :5], axis=1) > 0, 1, 0)
        a0_top_5 = np.where(np.count_nonzero(near_anomalies0[:, :5], axis=1) > 0, 1, 0)
        a1_top_5 = np.where(np.count_nonzero(near_anomalies1[:, :5], axis=1) > 0, 1, 0)
        a2_top_5 = np.where(np.count_nonzero(near_anomalies2[:, :5], axis=1) > 0, 1, 0)
        a3_top_5 = np.where(np.count_nonzero(near_anomalies3[:, :5], axis=1) > 0, 1, 0)
        a4_top_5 = np.where(np.count_nonzero(near_anomalies4[:, :5], axis=1) > 0, 1, 0)
        a5_top_5 = np.where(np.count_nonzero(near_anomalies5[:, :5], axis=1) > 0, 1, 0)
        a6_top_5 = np.where(np.count_nonzero(near_anomalies6[:, :5], axis=1) > 0, 1, 0)
        a7_top_5 = np.where(np.count_nonzero(near_anomalies7[:, :5], axis=1) > 0, 1, 0)
        a8_top_5 = np.where(np.count_nonzero(near_anomalies8[:, :5], axis=1) > 0, 1, 0)

        # Avg distance to abnormal instances
        a = np.mean(self.distances[:, self.anomalies], axis=1) if len(self.anomalies) > 0 else np.zeros(self.size)
        a_min = np.min(self.distances[:, self.anomalies], axis=1) if len(self.anomalies) > 0 else np.zeros(self.size)
        a0 = np.mean(self.distances[:, self.anomalies0], axis=1) if len(self.anomalies0) > 0 else np.zeros(self.size)
        a0_min = np.min(self.distances[:, self.anomalies0], axis=1) if len(self.anomalies0) > 0 else np.zeros(self.size)
        a1 = np.mean(self.distances[:, self.anomalies1], axis=1) if len(self.anomalies1) > 0 else np.zeros(self.size)
        a1_min = np.min(self.distances[:, self.anomalies1], axis=1) if len(self.anomalies1) > 0 else np.zeros(self.size)
        a2 = np.mean(self.distances[:, self.anomalies2], axis=1) if len(self.anomalies2) > 0 else np.zeros(self.size)
        a2_min = np.min(self.distances[:, self.anomalies2], axis=1) if len(self.anomalies2) > 0 else np.zeros(self.size)
        a3 = np.mean(self.distances[:, self.anomalies3], axis=1) if len(self.anomalies3) > 0 else np.zeros(self.size)
        a3_min = np.min(self.distances[:, self.anomalies3], axis=1) if len(self.anomalies3) > 0 else np.zeros(self.size)
        a4 = np.mean(self.distances[:, self.anomalies4], axis=1) if len(self.anomalies4) > 0 else np.zeros(self.size)
        a4_min = np.min(self.distances[:, self.anomalies4], axis=1) if len(self.anomalies4) > 0 else np.zeros(self.size)
        a5 = np.mean(self.distances[:, self.anomalies5], axis=1) if len(self.anomalies5) > 0 else np.zeros(self.size)
        a5_min = np.min(self.distances[:, self.anomalies5], axis=1) if len(self.anomalies5) > 0 else np.zeros(self.size)
        a6 = np.mean(self.distances[:, self.anomalies6], axis=1) if len(self.anomalies6) > 0 else np.zeros(self.size)
        a6_min = np.min(self.distances[:, self.anomalies6], axis=1) if len(self.anomalies6) > 0 else np.zeros(self.size)
        a7 = np.mean(self.distances[:, self.anomalies7], axis=1) if len(self.anomalies7) > 0 else np.zeros(self.size)
        a7_min = np.min(self.distances[:, self.anomalies7], axis=1) if len(self.anomalies7) > 0 else np.zeros(self.size)
        a8 = np.mean(self.distances[:, self.anomalies8], axis=1) if len(self.anomalies8) > 0 else np.zeros(self.size)
        a8_min = np.min(self.distances[:, self.anomalies8], axis=1) if len(self.anomalies8) > 0 else np.zeros(self.size)
        # Avg distance to normal instances
        n = np.mean(self.distances[:, self.normalies], axis=1) if len(self.normalies) > 0 else np.zeros(self.size)
        n_min = np.min(self.distances[:, self.normalies], axis=1) if len(self.normalies) > 0 else np.zeros(self.size)

        c = self._scores
        features = np.concatenate((
            np.expand_dims(a0_top_5, axis=1),
            np.expand_dims(a1_top_5, axis=1),
            np.expand_dims(a2_top_5, axis=1),
            np.expand_dims(a3_top_5, axis=1),
            np.expand_dims(a4_top_5, axis=1),
            np.expand_dims(a5_top_5, axis=1),
            np.expand_dims(a6_top_5, axis=1),
            np.expand_dims(a7_top_5, axis=1),
            np.expand_dims(a8_top_5, axis=1),
            np.expand_dims(a0, axis=1),
            np.expand_dims(a0_min, axis=1),
            np.expand_dims(a1, axis=1),
            np.expand_dims(a1_min, axis=1),
            np.expand_dims(a2, axis=1),
            np.expand_dims(a2_min, axis=1),
            np.expand_dims(a3, axis=1),
            np.expand_dims(a3_min, axis=1),
            np.expand_dims(a4, axis=1),
            np.expand_dims(a4_min, axis=1),
            np.expand_dims(a5, axis=1),
            np.expand_dims(a5_min, axis=1),
            np.expand_dims(a6, axis=1),
            np.expand_dims(a6_min, axis=1),
            np.expand_dims(a7, axis=1),
            np.expand_dims(a7_min, axis=1),
            np.expand_dims(a8, axis=1),
            np.expand_dims(a8_min, axis=1),
            np.expand_dims(n, axis=1),
            np.expand_dims(n_min, axis=1),
            c),
            axis=1)

        return features


def make_train_env(datapaths):
    if len(datapaths) > 1:
        env = EnsembleTrainEnv(datapaths)
    else:
        env = TrainEnv(datapaths[0])
    env = bench.Monitor(env, logger.get_dir())
    return env


def make_eval_env(datapath, budget):
    env = EvalEnv(datapath, budget)
    return env


