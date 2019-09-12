import random as rand


class NeuralTemp:

    class Neuron:

        def __init__(self, layer, id, nbgvs = 0):
            self.layer = layer
            self.id = id
            self.value = 0.5
            self.weights = []
            self.weighted = []
            self.bias = rand.uniform(nbgvs * -1, nbgvs)

    class Weight:

        def __init__(self, w_neuron, value):
            self.w_neuron = w_neuron
            self.value = value

    class Network:

        def __init__(self, nt):
            self.neuron = []
            self.nt = nt

            self.subjectivity = rand.uniform(0, 1) + 1
            self.ngs = rand.randint(0, nt.output_neuron_amt)
            self.nrs = rand.randint(0, nt.output_neuron_amt)
            self.wgs = rand.randint(0, nt.output_neuron_amt)
            self.wrs = rand.randint(0, nt.output_neuron_amt)
            self.nbgs = rand.randint(0, nt.output_neuron_amt)
            self.nbrs = rand.randint(0, nt.output_neuron_amt)
            self.wgvs = rand.uniform(0, 1)
            self.nbgvs = rand.uniform(0, 1)
            self.nbvs = rand.uniform(0, 1)
            self.wvs = rand.uniform(0, 1)

            self.fitness = 0

            for i in range(0, nt.layer_amt):
                self.neuron.append([])

            for i in range(0, nt.input_neuron_amt):
                self.neuron[0].append(nt.Neuron(0, i))

            for i in range(0, nt.output_neuron_amt):
                self.neuron[nt.output_layer].append(nt.Neuron(nt.output_layer, i, self.nbgvs))

        def add_weight(self, neuron, w_neuron, value):
            neuron.weights.append(self.nt.Weight(w_neuron, value))
            w_neuron.weighted.append(neuron)

        def generate_weight(self):
            neuron_list = self.get_neuron_list(0, self.nt.layer_amt)
            neuron = neuron_list[rand.randint(self.nt.input_neuron_amt, len(neuron_list) - 1)]
            w_neuron_list = self.get_neuron_list(0, neuron.layer)
            w_neuron = w_neuron_list[rand.randint(0, len(w_neuron_list) - 1)]

            value = rand.uniform(self.wgvs * -1, self.wgvs)
            self.add_weight(neuron, w_neuron, value)

        def remove_weight(self):
            neuron_list = self.get_neuron_list(1, self.nt.layer_amt)
            neuron = neuron_list[rand.randint(0, len(neuron_list) - 1)]

            while len(neuron.weights) == 0:
                if self.has_no_weights():
                    return
                neuron = neuron_list[rand.randint(0, len(neuron_list) - 1)]

            weight_index = rand.randint(0, len(neuron.weights) - 1)
            weight = neuron.weights[weight_index]
            weight.w_neuron.weighted.remove(neuron)
            neuron.weights.pop(weight_index)

        def generate_neuron(self):
            if len(self.get_neuron_list(0, self.nt.layer_amt)) == self.nt.input_neuron_amt + self.nt.max_h_neuron_amt * self.nt.max_hidden_layer_amt + self.nt.output_neuron_amt:
                return
            layer = rand.randint(1, self.nt.max_hidden_layer_amt)
            while len(self.neuron[layer]) == self.nt.max_h_neuron_amt:
                layer = rand.randint(1, self.nt.max_hidden_layer_amt)

            self.neuron[layer].append(self.nt.Neuron(layer, len(self.neuron[layer]), self.nbgvs))

        def remove_neuron(self):
            if len(self.get_neuron_list(1, self.nt.output_layer)) == 0:
                return
            layer = rand.randint(1, self.nt.max_hidden_layer_amt)
            while len(self.neuron[layer]) == 0:
                layer = rand.randint(1, self.nt.max_hidden_layer_amt)
            id = rand.randint(0, len(self.neuron[layer]) - 1)

            for weight in self.neuron[layer][id].weights:
                for i in range(0, len(weight.w_neuron.weighted)):
                    if weight.w_neuron.weighted[i] == self.neuron[layer][id]:
                        weight.w_neuron.weighted.pop(i)
                        break

            for weighted in self.neuron[layer][id].weighted:
                for i in range(0, len(weighted.weights)):
                    if weighted.weights[i].w_neuron == self.neuron[layer][id]:
                        weighted.weights.pop(i)
                        break

            self.neuron[layer].pop(id)

            for i in range(id, len(self.neuron[layer])):
                self.neuron[layer][i].id -= 1

        def generate_bias(self):
            neuron_list = self.get_neuron_list(1, self.nt.layer_amt)
            neur = rand.randint(0, len(neuron_list) - 1)
            neuron_list[neur].bias = rand.uniform()

        def get_neuron_list(self, start_layer, end_layer):
            neuron_list = []
            for i in range(start_layer, end_layer):
                for neuron in self.neuron[i]:
                    neuron_list.append(neuron)
            return neuron_list

        def has_no_weights(self):
            for i in range(1, self.nt.layer_amt):
                for neuron in self.neuron[i]:
                    if len(neuron.weights) != 0:
                        return False
            return True

    def __init__(self, pop_size, input_neuron_amt, max_h_neuron_amt, output_neuron_amt, max_hidden_layer_amt, fitness_function=0):
        self.pop_size = pop_size
        self.input_neuron_amt = input_neuron_amt
        self.max_h_neuron_amt = max_h_neuron_amt
        self.output_neuron_amt = output_neuron_amt
        self.max_hidden_layer_amt = max_hidden_layer_amt
        if fitness_function == 0:
            self.fitness_function = self.evaluate_fitnesses
        else:
            self.fitness_function = fitness_function

        self.cost_degree = 1

        self.layer_amt = max_hidden_layer_amt + 2
        self.output_layer = max_hidden_layer_amt + 1
        self.inputs = []
        self.desired_outputs = []

        self.network = []

        self.neuron_cap = 0.999999
        self.neuron_floor = 0.000001
        self.weight_cap = 10
        self.weight_floor = -10

        self.create_networks()
        self.initialize_networks()

    def create_networks(self):
        for i in range(0, self.pop_size):
            self.network.append(self.Network(self))

    def initialize_networks(self):
        for network in self.network:
            for i in range(0, self.output_layer):
                if rand.randint(0, self.output_neuron_amt) < network.wgs:
                    network.generate_weight()

    def add_input_output(self, input_list, output_list):
        self.inputs.append(input_list)
        self.desired_outputs.append(output_list)

    def input_data(self, network, input_list):
        for i in range(0, self.input_neuron_amt):
            network.neuron[0][i].value = input_list[i]

        for i in range(1, self.layer_amt):
            for j in range(0, len(network.neuron[i])):
                weight_total = 0
                for weight in network.neuron[i][j].weights:
                    weight_total += weight.w_neuron.value * weight.value
                weight_total += network.neuron[i][j].bias
                try:
                    network.neuron[i][j].value = 1 / (1 + 2.718 ** (weight_total*(-1)))
                except OverflowError:
                    if weight_total > 0:
                        network.neuron[i][j].value = 1.0
                    else:
                        network.neuron[i][j].value = 0.0

                if network.neuron[i][j].value < self.neuron_floor:
                    network.neuron[i][j].value = 0
                elif network.neuron[i][j].value > self.neuron_cap:
                    network.neuron[i][j].value = 1

    def evaluate_fitnesses(self, nt):
        for network in self.network:
            network.fitness = 0
            for i in range(0, len(self.inputs)):
                self.input_data(network, self.inputs[i])
                for j in range(0, self.output_neuron_amt):
                    network.fitness += abs(self.desired_outputs[i][j] - network.neuron[self.output_layer][j].value) ** self.cost_degree

    def produce_child(self, parent):
        child = self.Network(self)

        # Subjectivities
        child.subjectivity = parent.subjectivity + rand.uniform(-1, 1)
        child.ngs = parent.ngs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.nrs = parent.nrs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.wgs = parent.wgs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.wrs = parent.wrs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.nbgs = parent.nbgs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.nbrs = parent.nbrs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.wgvs = parent.wgvs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.nbgvs = parent.nbgvs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.nbvs = parent.nbvs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)
        child.wvs = parent.wvs + rand.uniform(parent.subjectivity * -1, parent.subjectivity)

        # Subjectivity floors

        if child.subjectivity < 1:
            child.subjectivity = 1
        if child.ngs < 0.1:
            child.ngs = 0.1
        if child.nrs < 0.1:
            child.nrs = 0.1
        if child.wgs < 0.1:
            child.wgs = 0.1
        if child.wrs < 0.1:
            child.wrs = 0.1
        if child.nbgs < 0.1:
            child.nbgs = 0.1
        if child.nbrs < 0.1:
            child.nbrs = 0.1
        if child.wgvs < 0.1:
            child.wgvs = 0.1
        if child.nbgvs < 0.1:
            child.nbgvs = 0.1
        if child.nbvs < 0.1:
            child.nbvs = 0.1
        if child.wvs < 0.1:
            child.wvs = 0.1

        # Clear neurons
        for i in range(1, self.layer_amt):
            child.neuron[i].clear()

        # Weights and biases
        for i in range(1, self.layer_amt):
            for j in range(0, len(parent.neuron[i])):
                child.neuron[i].append(self.Neuron(i, j))
                child.neuron[i][j].bias = parent.neuron[i][j].bias + rand.uniform(parent.nbvs * -1, parent.nbvs)
                for weight in parent.neuron[i][j].weights:
                    weight_value = weight.value + rand.uniform(parent.wvs * -1, parent.wvs)
                    if weight_value < self.weight_floor:
                        weight_value = self.weight_floor
                    elif weight_value > self.weight_cap:
                        weight_value = self.weight_cap
                    child.add_weight(child.neuron[i][j], child.neuron[weight.w_neuron.layer][weight.w_neuron.id], weight_value)

        # Mutations
        try:
            weight_gen_amt = rand.randint(0, int(parent.wgs))
        except ValueError:
            weight_gen_amt = 0
        try:
            weight_rem_amt = rand.randint(0, int(parent.wrs))
        except ValueError:
            weight_rem_amt = 0
        try:
            neuron_gen_amt = rand.randint(0, int(parent.ngs))
        except ValueError:
            neuron_gen_amt = 0
        try:
            neuron_rem_amt = rand.randint(0, int(parent.wrs))
        except ValueError:
            neuron_rem_amt = 0

        if self.max_hidden_layer_amt != 0:
            for i in range(1, neuron_rem_amt):
                child.remove_neuron()
            for i in range(1, neuron_gen_amt):
                child.generate_neuron()

        for i in range(1, weight_rem_amt):
            child.remove_weight()
        for i in range(1, weight_gen_amt):
            child.generate_weight()

        return child

    def reproduce(self):

        self.fitness_function(self)

        self.network = sorted(self.network, key=lambda net: net.fitness)

        for i in range(0, int(self.pop_size / 2)):
            self.network[i + int(self.pop_size / 2)] = self.produce_child(self.network[i])

    def auto_reproduce(self, fitness_threshold=1, ensure_amt=10, log_interval=1):

        ended = False

        log_counter = 1
        failure_cap = 10
        success_cap = 5
        failures = 0
        successes = 0
        failure_threshold = 0.5
        last_fitness = -1

        while not ended:
            self.reproduce()

            if log_counter % log_interval == 0:
                print(self.network[0].fitness)

            log_counter += 1

            if self.network[0].fitness < 1 or self.network[0].fitness < fitness_threshold:
                if self.network[0].fitness < fitness_threshold and self.cost_degree == 1:
                    ended = True
                elif self.network[0].fitness < fitness_threshold / 2:
                    print("Reproducing " + str(ensure_amt) + " times for ensurance...")
                    for i in range(0, ensure_amt):
                        self.reproduce()
                    self.cost_degree -= 1
                    print("Cost degree: " + str(self.cost_degree))
                    failures = 0

            if last_fitness - self.network[0].fitness < failure_threshold:
                failures += 1
                successes = 0
            else:
                failures = 0
                successes += 1
                last_fitness = self.network[0].fitness
            if failures >= failure_cap:
                self.cost_degree += 1
                print("Cost degree: " + str(self.cost_degree))
                failures = 0
            elif successes >= success_cap and self.cost_degree != 1:
                self.cost_degree -= 1
                print("Cost degree: " + str(self.cost_degree))
                successes = 0

    def reproduce_until(self, max_fitness):
        self.reproduce()
        while self.network[0].fitness > max_fitness:
            self.reproduce()