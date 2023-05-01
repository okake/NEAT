from genome_cartpole import DefaultGenome
from speciation import DefaultSpeciesSet
from feedforward_cartpole import FeedForwardNetwork
from reproduction import DefaultReproduction
import numpy as np
from matplotlib import pyplot
import math
import gym
from gym.wrappers import RecordVideo


pop_size = 50
generation = 0

num_inputs = 4
num_outputs = 2

best_genome = None

best_fitness_hist = []
avarage_fitness_hist = []

population = DefaultGenome.create_new(DefaultGenome, pop_size,num_inputs,num_outputs)
species = DefaultSpeciesSet()

while generation != 1000:
    species.speciate(population,generation)
    #print(species.species)
    FeedForwardNetwork.eval_genomes(list(population.items()),num_inputs,num_outputs)
    
    best = None
    fitness_sum = 0
    for g in population.values():
        fitness_sum += g.fitness
        if best is None or best.fitness < g.fitness:
            best = g
            
    if best_genome is None or best.fitness > best_genome.fitness:
        best_genome = best

    best_fitness_hist.append(best_genome.fitness)
    avarage_fitness_hist.append(fitness_sum/pop_size)

    """
    print(generation)
    print('\nBest genome:\n{!s}'.format(best_genome))
    print('\nOutput:')
    """
        
    fv = max(g.fitness for g in population.values())
    #print(fv)
    if generation > 100:
        print(sum(best_fitness_hist[-100:])/100)
    if sum(best_fitness_hist[-100:])/100 > 475 and generation > 100:
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(generation, best.size()))
        break
        
    popus = DefaultReproduction(num_inputs,num_outputs)
    population = popus.reproduce(species,pop_size,generation)
    
    generation += 1

winner = best_genome
print('\nBest genome:\n{!s}'.format(winner))
print('\nOutput:')
winner_net = FeedForwardNetwork.create(winner,num_inputs,num_outputs)
env = RecordVideo(gym.make("CartPole-v1"), "cartpole_pop5")
observation = env.reset()
for t in range(500):
    env.render()
    action = np.array(winner_net.activate(observation.tolist()))
    #2softmax
    action_exp = np.exp(action)
    sum_exp = np.sum(action_exp)
    action = action_exp/sum_exp
    action = np.argmax(action)
    #--------
    observation, reward, done, info = env.step(action)
    if done:
        break
env.close()

g1 = pyplot.plot(avarage_fitness_hist, color = "red")
g2 = pyplot.plot(best_fitness_hist)
pyplot.legend((g1[0], g2[0]), ("Average", "Best"), loc=4)
pyplot.show()