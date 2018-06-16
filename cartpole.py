# Imports
import gym
import numpy as np
import pandas as pd
import time
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense


from generic_genetic import neuroevolution

class pole_dancer:
    
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.input_dims = self.env.observation_space.shape[0]
        self.output_dims = self.env.action_space.n
        self.population_size = 55

        self.trainer = neuroevolution(
            population_size = self.population_size, 
            layers = [self.input_dims, 8, self.output_dims], 
            mutation_rate = 0.20, 
            n_best_survivors = 5, 
            n_total_survivors = 10, 
            crossover_rate = 0.50
        )
        
        self.population = self.trainer.init_generation()
        self.fitness = np.full(self.trainer.population_size, float('-inf'))
        self.cur_gen = 0
        self.TIME_STEPS = 10000

    def simulate(self, i):
        individual = self.population[i]
        observation = self.env.reset()
        reward = 0
        done = False

        for _ in range(0, self.TIME_STEPS):
            #self.env.render()
            action = individual.predict(np.reshape(observation, (1, 4)), 1)
            observation, reward_state, done, info = self.env.step(np.argmax(action))
            reward += reward_state
            if done:
                break
        
        self.fitness[i] = reward
    
    def progress_step(self):
        for i in range(0, self.population_size):
            self.simulate(i)
        self.population = self.trainer.next_generation(self.fitness)
        print(max(self.fitness))
        #self.save()

    def play_god(self, generations):
        for _ in range(0, generations):
            self.cur_gen += 1
            print("GENERATION ", self.cur_gen)
            self.progress_step()
            if (self.cur_gen % 10 == 0):
                self.save()

    def save(self):
        i = 0
        for individual in self.population:
            individual.save_weights("current_pool/individual_" + str(i) + ".keras")
            i += 1

    def load(self):
        for i in range(self.population_size):
            self.population[i].load_weights("current_pool/individual_" + str(i) + ".keras")

def main():
    world = pole_dancer()
    world.play_god(20)

if __name__ == "__main__":
    main()