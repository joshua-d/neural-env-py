# neural_temp.py
A python module for creating neural networks and conducting machine learning using NEAT.

This is the new and improved python translation of NeuralTemp.h, the C++ header.

Setting up your neural network environment with neural_temp.py is simple:

```
import neural_temp

nt = neural_temp.NeuralTemp(
  
    Population size,
    Amount of neurons in the input layer,
    Maximum amount of neurons to be generated in the hidden layers,
    Amount of neurons in the output layer,
    Amount of inputs in training data
    
  )
  ```
  
Here, a new instance of NeuralTemp has been constructed, and a population of neural networks have been generated with random properties fitting the arguments provided.

Next, designate the inputs of each member of the population to the inputs in the dataset. Each input has a corresponding desired output that the members will be judged based on.

For a data set of two input lists: `data1` and `data2`, with desired output lists: `output1` and `output2`, respectively:

```
nt.add_input_output(data1, output1)
nt.add_input_output(data2, output2)
```

The networks are ready to be reproduced to develop a neural network that can produce the desired results of the data set.

```
nt.auto_reproduce()
```

When reproduction is finished, nt.network[0] points to the best-performing neural network. To test, input the values of a data list into the input neurons of the network:

```
nt.network[0].input_data(data1)
```

The list `nt.network[0].neuron[nt.output_layer]` holds the values of the neural network's output neurons, and they should be close to the values of the desired output for the inputted data, in this case, output1.
  
This list can be obtained using `nt.get_best_network().get_output_neurons()`.
