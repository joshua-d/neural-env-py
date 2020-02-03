# neural_env.py
A python module for creating neural networks and conducting machine learning using NEAT.

![](Pong/Pong.gif)

*NN learning to play Pong using neural_env.py*

*Input neurons: ball x-position, ball y-position, ball x-velocity, ball y-velocity, paddle y-position*

*Notice how after only 10 minutes of learning, the NN starts to learn how to use the ball's angle to predict where the paddle should be, rather than just matching the y position of the paddle with the y position of the ball.*

<br>

### Contents
- [Overview](#overview)
- [Basic Usage](#usage)

<br>
<a name="overview"/>

## Overview

The neural_env module provides tools for easy setup of a machine learning environment based on neuroevolution of augmenting topologies, or NEAT.

NEAT involves using natural selection, or survival of the fittest, to facilitate evolution of artificial neural networks in favor of producing a desired output when given a specific input.

<br>

The process of training a neural network to produce a desired output when it is given a specific input is as follows:
1) Begin with a population of randomly generated neural networks with different genes and characteristics.
2) Evaulate each neural network's performance (fitness) by comparing its output to the desired output.
3) Allow the neural networks that performed the best (have the best fitness) to reproduce, producing new neural networks of slightly different characteristics and genes (based on the genes of the parents). These new neural networks replace those with worse fitness.
4) Repeat fitness evaulation and reproduction until a neural network of desired performance is produced.

<br>

Upon construction of an instance of the NeuralEnv class, the user determines universal characteristics of the environment that all neural networks share, including:
- Population size (number of different neural networks to compete)
- Input neuron amount
- Maximum hidden layer neuron amount
- Output neuron amount
- Number of hidden layers

The construction will generate the intial population of neural networks based on the universal characteristics specified by the user.

<br>

The user has 3 options for evaluation of the neural networks' fitnesses:

**Method 1)** Specify a set of inputs and a set of respective desired outputs that the neural networks will be evaluated based upon

**Method 2)** Specify a user-defined fitness function that will be called on each neural network to evaluate fitness prior to reproduction

**Method 3)** Do not specify either a fitness function or a set of inputs and desired outputs; manually evaulate the neural networks prior to reproduction

More details provided in ***Basic Usage***.

<br>

After specification of a fitness evalutation method, the environment is ready for reproduction.

<br>
<a name="usage"/>

## Basic Usage

Setting up your environment with neural_env.py is simple:

```
import neural_env


# Create an environment instance and set universal characteristics

nenv = neural_env.NeuralEnv(
  
    Population size (pop_size),
    Amount of neurons in the input layer (input_neuron_amt),
    Maximum amount of neurons to be generated in the hidden layers (max_h_neuron_amt),
    Amount of neurons in the output layer (output_neuron_amt),
    Number of hidden layers (max_hidden_layer_amt),
    Fitness function [optional] (fitness_function)
    
  )
  ```
  
Here, a new instance of NeuralEnv has been constructed, and a population of neural networks have been generated with random properties fitting the arguments provided.

<br>

#### Fitness Evaluation Method 1

Designate the set of inputs and the set of corresponding desired outputs that the networks will be evaluated based upon.

An input is a set of **input neuron values** that will be given to a network.
A desired output is a set of **output neuron values** that a network's output neuron values will be tested against.

```
nenv = neural_env.NeuralEnv(pop_size = 1000, input_neuron_amt = 4, max_h_neuron_amt = 0, output_neuron_amt = 2, max_hidden_layer_amt = 0)

input_1 = [1, 0, 1, 0]
input_2 = [1, 1, 1, 0]

desired_output_1 = [1, 0]
desired_output_2 = [0, 1]

nenv.add_input_output(input1, desired_output_1)
nenv.add_input_output(input2, desired_output_2)
```
In this example, neural networks will be evaluated based on how close their output neuron values are to the desired output values when their input neuron values are set to the input values.

Once at least one input-desired output is defined, the networks are ready for reproduction.

```
nenv.reproduce()
```

The `reproduce()` function will compare each network's output values to the desired output values when given each input, and assign a resulting fitness. The **closer** the fitness to **zero**, the better the performance of the network. It will then reproduce based on the evaluated fitnesses.

<br>

#### Fitness Evaulation Method 2

A valid fitness function takes one neural network as an argument and computes a fitness for said neural network. The **better** the performance, the **lesser** the fitness value.

```
def do_realtime_task(network):
  # Use this neural network to perform a task such as playing a video game
  return score

def example_fitness_function(network):
  score = do_realtime_task(network)
  network.fitness = -score
  
```
In this example, the network is used to perform a task and given a score. The network's fitness is set to `-score` so that higher scores result in better fitness.

If a fitness function is provided to the NeuralEnv constructor, the networks are ready for reproduction.

```
nenv.reproduce()
```

The `reproduce()` function will call the specified fitness function on each neural network in the population and then reproduce based on the evaluated fitnesses.

<br>

#### Fitness Evaluation Method 3

If no fitness function or inputs/desired outputs are specified, the `reproduce` function will simply reproduce based on the current `fitness` of each network. This method gives the user the most power, but it is his or her responsibility to set each network's `fitness` value prior to each reproduction.

```
def play_pong(network_1, network_2):
  # These 2 networks play against each other
  network_1.fitness = -n1_score
  network_2.fitness = -n2_score
  
 
# Each neural network is part of a game and is assigned a fitness
i = 0
while i < nenv.pop_size:
  play_pong(nenv.networks[i], nenv.networks[i + 1])
  i += 2

nenv.reproduce()

```

<br>

#### Reproduction

When reproduction is finished, nenv.networks[0] points to the best-performing neural network. To test, input the values of a data list into the input neurons of the network:

```
nenv.network[0].input_data(data1)
```

The list `nenv.network[0].neuron[nenv.output_layer]` holds the values of the neural network's output neurons, and they should be close to the values of the desired output for the inputted data, in this case, output1.
  
This list can be obtained using `nenv.get_best_network().get_output_neurons()`.
