import neural_env


def main():
    nenv = neural_env.NeuralEnv(1000, 2, 2, 1, 1)

    input_1 = [0, 0]
    input_2 = [0, 1]
    input_3 = [1, 0]
    input_4 = [1, 1]

    desired_output_1 = [0]
    desired_output_2 = [1]
    desired_output_3 = [1]
    desired_output_4 = [0]

    nenv.add_input_output(input_1, desired_output_1)
    nenv.add_input_output(input_2, desired_output_2)
    nenv.add_input_output(input_3, desired_output_3)
    nenv.add_input_output(input_4, desired_output_4)

    nenv.auto_reproduce(0.2)

    inputs = [input_1, input_2, input_3, input_4]

    for i in range(len(inputs)):
        nenv.get_best_network().input_data(inputs[i])
        print("Output " + str(i + 1) + ":")
        for o_n in nenv.get_best_network().get_output_neurons():
            print(o_n.value)
        print("")


main()