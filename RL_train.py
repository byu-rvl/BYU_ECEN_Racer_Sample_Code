# Note, much of this code was from the CS Deep Learning class taught at BYU that Drew Sumsion took
# Drew Sumsion is writing this and acknowledging the help and gratitude for offering this code for me to build off of.
from datetime import datetime
from pathlib import Path
from simulation import Simulator
import gc
import torch
import torch.nn as nn
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = []
        for d in data:
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class RLTraining:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        # TODO adjust these hyper parameters as desired
        self.lr = 1e-5
        self.epochs = 1000
        self.saveEvery_epochs = 10
        self.iterTrainSamples = 5
        self.iterTestEvaluations = 20
        self.gamma = 0.9
        self.epsilon = 0.2
        self.policy_epochs = 5
        self.MAX_REWARD = 3000
        self.testReward, testScore = 0, 0
        self.networkOutputSize = 2 #This will depend on how many output parameters your network returns

        self.policy_network = None  # TODO define policy network
        self.value_network = None  # TODO define value network
        self.optim = torch.optim.Adam(chain(self.policy_network.parameters(), self.value_network.parameters()),
                                      lr=self.lr)

        # Init environment TODO Adjust the mapParameters, carParameters and cameraSettings as desired. These are the
        #                       parameters you may want to vary through out training to ensure your model generalizes
        #                       to be applicable in the real world.
        self.mapParameters = {
            "loops": 2,
            "size": (10, 10),
            "expansions": 40,
            "complications": 2
        }
        self.carParameters = {
            "wheelbase": 8,  # inches, influences how quickly the steering will turn the car.  Larger = slower
            "maxSteering": 30.0,  # degrees, extreme (+ and -) values of steering
            "steeringOffset": 0.0,  # degrees, since the car is rarely perfectly aligned
            "minVelocity": 0.0,  # pixels/second, slower than this doesn't move at all.
            "maxVelocity": 480.0,
            # pixels/second, 8 pixels/inch, so if the car can move 5 fps that gives us 480 pixels/s top speed
        }

        resolution = (1920, 1080)  # TODO change resolution to desired size, you may want this to be higher than your
        #                               network accepts so that it doesn't loose the smaller obstacles initially.
        #                               However the smaller this is the faster it will run.
        cameraSettings = {
            "resolution": (resolution[0], resolution[1]),
            "fov": {"diagonal": 94},  # realsense diagonal fov is 94 degrees IIRC
            "angle": {"roll": 0, "pitch": 0, "yaw": 0},
            # don't go too crazy with these, my code should be good up to like... 45 degrees probably? But the math
            # gets unstable
            "height": 66  # 8 pixels/inch - represents how high up the camera is relative to the road
        }
        self.env = Simulator(cameraSettings)

        # TODO adjust the testRealParameters and testCameraSettings.  These are used in the evaluation of the model.
        #  Likely you will want this to be as close to the real vehicle as possible. These values below may be close enough.
        self.testRealParameters = {
            "wheelbase": 8,  # inches, influences how quickly the steering will turn the car.  Larger = slower
            "maxSteering": 30.0,  # degrees, extreme (+ and -) values of steering
            "steeringOffset": 0.0,  # degrees, since the car is rarely perfectly aligned
            "minVelocity": 0.0,  # pixels/second, slower than this doesn't move at all.
            "maxVelocity": 480.0,
            # pixels/second, 8 pixels/inch, so if the car can move 5 fps that gives us 480 pixels/s top speed
        }
        testCameraSettings = {
            "resolution": (resolution[0], resolution[1]),
            "fov": {"diagonal": 94},  # realsense diagonal fov is 94 degrees IIRC
            "angle": {"roll": 0, "pitch": 0, "yaw": 0},
            # don't go too crazy with these, my code should be good up to like... 45 degrees probably? But the math
            # gets unstable
            "height": 66  # 8 pixels/inch - represents how high up the camera is relative to the road
        }
        self.testEnv = Simulator(testCameraSettings)

        here = Path(__file__).resolve()
        saveDir = str(datetime.now())
        saveDir = saveDir.replace(" ", "__")
        saveDir = saveDir.replace("-", "_")
        saveDir = saveDir.replace(":", "_")
        saveDir = saveDir.replace(".", "_")
        self.savePath = here / "RL_results" / saveDir
        self.savePath.mkdir(exist_ok=True, parents=True)

    def calculate_return(self, memory, rollout):
        """Return memory with calculated return in experience tuple

          Args:
              memory (list): (state, action, action_dist, return) tuples
              rollout (list): (state, action, action_dist, reward) tuples from last rollout

          Returns:
              list: memory updated with (state, action, action_dist, return) tuples from rollout
        """
        firstTime = True
        previous = None
        newMem = []
        newMem1 = []
        for newState, newAngle, newSpeed, newAction_dist, reward in reversed(rollout):

            if firstTime:
                new_return = reward
                firstTime = False
            else:
                new_return = reward + previous * self.gamma
            previous = new_return
            newMem.append([newState, newAngle, newSpeed, newAction_dist, new_return])

        for x in reversed(newMem):
            newMem1.append(x)
        memory.append(newMem1)
        return memory

    def get_action_ppo(self, state):
        """Sample action from the distribution obtained from the policy network

          Args:
              state (np-array): current state, size (state_size)

          Returns:
              int: angle sampled from output distribution of policy network
              int: speed sampled from output distribution of policy network
              array: output distribution of policy network
        """
        # run the state through the network
        state_tensor = torch.Tensor(np.array([state])).to(self.device)
        policy_output = self.policy_network(state_tensor).to(self.device)
        angle, speed = None, None  # TODO determine action based on network output(s). You may want to to set speed to a constant value
        return angle, speed, policy_output

    def preprocessImage(self, image):
        """
        Do any preprocessing of the image returned by the simulation here
        """
        # TODO preprocess if desired

        return None

    def learn_ppo(self, memory_dataloader):
        """Implement PPO policy and value network updates. Iterate over your entire
           memory the number of times indicated by policy_epochs.

          Args:
              memory_dataloader (DataLoader): dataloader with (state, action, action_dist, return, discounted_sum_rew) tensors
        """
        self.policy_network.reset()
        for epoch in range(0, self.policy_epochs):
            for set in memory_dataloader:
                self.optim.zero_grad()
                loss = None
                for state, action, action_dist, theReturn in set:

                    # first set ups
                    state = state.float().to(self.device)
                    theReturn = theReturn.float().to(self.device)
                    action = action.to(self.device)
                    action_dist = action_dist.to(self.device)
                    action_dist = action_dist.detach()

                    # get the value loss
                    # value.reset()
                    test = self.value_network(state).squeeze()
                    # value.deleteHidden()
                    gc.collect()
                    torch.cuda.empty_cache()
                    squeezed = torch.argmax(test, dim=0)
                    value_loss = nn.functional.mse_loss(squeezed, theReturn)

                    # advantage for the policy loss
                    advantage = theReturn - squeezed  # actual - expected
                    advantage = advantage.detach()

                    # policy loss
                    encoded = nn.functional.one_hot(action,
                                                    num_classes=self.networkOutputSize).bool().squeeze()
                    policy_ratio = self.policy_network(state)[encoded] / action_dist.squeeze()[encoded]

                    # prevent overfitting
                    clip_policy_ratio = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon)
                    policy_loss = -1 * torch.mean(
                        torch.minimum(policy_ratio * advantage, clip_policy_ratio * advantage))

                    # combine the loss
                    if loss is None:
                        loss = value_loss + policy_loss
                    else:
                        loss += value_loss + policy_loss

                    # normal dep learning things
                loss.backward()
                self.optim.step()
                self.policy_network.reset()
                self.value_network.reset()

    def evaluateOnReal(self):
        """
        This function tests your model on the simulated version of the course in the class. It evaluates and
        """

        cum_rewards = []
        score = 0
        self.policy_network.eval()
        with torch.no_grad():
            for eval in range(self.iterTestEvaluations):
                random.seed(eval)
                state = self.testEnv.start(mapSeed="real", carParameters=self.testRealParameters)
                self.policy_network.reset()
                state = self.preprocessImage(state)
                done = False
                rewards = 0
                while not done and rewards < self.MAX_REWARD:
                    angle, speed, _ = self.get_action_ppo(state)
                    state = self.testEnv.step(angle, speed,display=False) #TODO if you would like to display course and camera view set display to True
                    reward,done = self.getReward(self.testEnv)
                    state = self.preprocessImage(state)
                    rewards += reward
                    score += 1
                cum_rewards.append(rewards)
        self.policy_network.train()
        return np.mean(cum_rewards), score / self.iterTestEvaluations

    def train(self):
        self.testRewards = []
        self.testScores = []
        self.trainScores = []
        self.trainRewards = []
        self.trainRewardsMean = []
        testReward = 0
        testScore = 0
        # Start main loop
        results_ppo = []
        results_ppo_mean = []
        loop = tqdm(total=self.epochs, position=0, leave=False)

        # create directory to save in


        constantSpeed = 100
        for epoch in range(self.epochs):

            memory = []  # Reset memory every epoch
            rewards = []  # Calculate average episodic reward per epoch
            scores = []
            # Begin experience loop
            for episode in range(self.iterTrainSamples):
                # Reset environment
                state = self.env.start(mapSeed=(episode + 1) * (epoch + 1), mapParameters=self.mapParameters,
                                       carParameters=self.carParameters)
                score = 0
                state = self.preprocessImage(state)
                done = False
                rollout = []
                cum_reward = 0  # Track cumulative reward

                self.policy_network.reset()

                # Begin episode
                while not done and cum_reward < self.MAX_REWARD:
                    # Get action
                    angle, speed, action_dist = self.get_action_ppo(state)
                    # Take step
                    next_state = self.env.step(angle, speed,display=False) #TODO if you would like to display course and camera view set display to True
                    reward, done = self.getReward(self.env)
                    # Store step
                    rollout.append((state, angle, speed, action_dist, reward))
                    score += 1
                    cum_reward += reward
                    state = self.preprocessImage(next_state)

                # Calculate returns and add episode to memory
                memory = self.calculate_return(memory, rollout)
                rewards.append(cum_reward)
                scores.append(score)
            # Train
            dataset = RLDataset(memory)
            loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
            self.policy_network.reset()

            # Print results
            self.trainRewards.extend(rewards)  # Store rewards for this epoch
            self.trainScores.append(np.mean(scores))
            self.trainRewardsMean.append(np.mean(rewards))
            loop.update(1)
            loop.set_description(
                "Epochs: {} Reward: {} TrainScore: {} TestReward: {} TestScore: {}".format(epoch, np.mean(rewards),
                                                                                           np.mean(scores), testReward,
                                                                                           testScore))

            if not (epoch % self.saveEvery_epochs):
                # save network parameters and plots of training and testing results every desired amount of epochs.
                # TODO you may want to also save if the average reward or score was extra high.
                testReward, testScore = self.evaluateOnReal()
                self.testRewards.append(testReward)
                self.testScores.append(testScore)
                self.saveResults(epoch)
            else:
                self.testRewards.append(testReward)
                self.testScores.append(testScore)
            self.learn_ppo(loader)
        return results_ppo
    def saveResults(self,epoch):
        policyPath = self.savePath / ("policy_epoch_" + str(epoch)  + ".pt")
        valuePath = self.savePath / ("value_epoch_" + str(epoch)  + ".pt")
        plotPath = self.savePath / ("plot_epoch_" + str(epoch) + ".png")
        plotMeanPath = self.savePath / ("plot_epoch_mean_" + str(epoch) + ".png")
        plotScorePath = self.savePath / ("plot_epoch_score_" + str(epoch) + ".png")

        torch.save(self.policy_network.state_dict(), str(policyPath))
        torch.save(self.value_network.state_dict(), str(valuePath))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        plt.plot(self.trainRewards)
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.savefig(plotPath)
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        plt.plot(self.trainRewardsMean, label="Train")
        plt.plot(self.testRewards, label="Test")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.savefig(plotMeanPath)
        plt.close()
        plt.cla()
        plt.clf()
        plt.plot(self.trainScores, label="Train")
        plt.plot(self.testScores, label="Test")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.savefig(plotScorePath)
    def getReward(self, env):
        # return negative reward if crashed positive reward if doing
        distToCenter, bearingOffset = env.getStats()
        # TODO create reward scheme
        return None,None


training = RLTraining()
training.train()
