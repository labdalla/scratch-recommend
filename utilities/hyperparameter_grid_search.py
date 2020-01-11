# Execute a hyperparameter grid search,
# to tune the hyperparamters of both the supervised and unsupervised models.
# Steps:
#   – for each hyperparameter, choose a few values of that hyperparameter that you want to experiment with.
#   - find all combinations of the hyperparameters [one value per hyperparameter]
#   - run the unsupervised and supervised models with each combination
#   - get the corresponding f1 score and add it to a big spreadsheet (to then examine later).

from fasttext import *
import itertools

SETTINGS = "./settings"
CBOW_MASTER_SETTINGS_FILE = os.path.abspath(os.path.join(SETTINGS, 'cbow_master_settings.txt'))
SKIPGRAM_MASTER_SETTINGS_FILE = os.path.abspath(os.path.join(SETTINGS, 'skipgram_master_settings.txt'))
SCRATCH_VECTORIZE = "../scratch-vectorize/"

# UNSUPERVISED HYPERPARAMETERS
MODEL_TYPE = ['model_type: cbow', 'model_type: skipgram']
MINCOUNT_UNSUPERVISED = ["minCount: 1", "minCount: 5"]
DIM = ["dim: 50", "dim: 64", "dim: 128", "dim: 175", "dim: 200"]
MINN_MAXNN = [("minn: 1", "maxn: 5"), ("minn: 1", "maxn: 8"), ("minn: 1", "maxn: 10")]
EPOCH_LR_UNSUPERVISED = [("epoch: 5", "lr: 0.1"), ("epoch: 10", "lr: 0.05"), ("epoch: 25", "lr: 0.01"), ("epoch: 50", "lr: 0.01")]


# SUPERVISED HYPERPARAMETERS
MINCOUNT_SUPERVISED = [1, 5]
EPOCH_LR_SUPERVISED = [(5, 0.1), (10, 0.1), (25, 0.05), (50, 0.05)]
WORD_NGRAMS = [1, 5, 10] # how many ngrams to look surrounding the current word

def get_combinations(hyperparameters):
    """
    Takes in all possible options for each hyperparameter, and returns all possible combinations between them.

    inputs:
        - hyperparamters: a list of lists, where each list corresponds to a certain hyperparameter and
                    contains the different options for that hyperparameter.

    returns: a list of all possible combinations of the different hyperparameters (which a specific value for each hyperparameter)
    """
    return list(itertools.product(*hyperparameters))

def add_hyperparameter(hyperparameter_string, hyperparamters_dictionary):
    """
    Takes in hyperparameter_string in the form of "hyperparameter_name: hyperparameter_value",
    and parses that, and adds it appropriately to hyperparamters_dictionary.

    inputs:
        - hyperparameter_string: in the form: "hyperparameter_name: hyperparameter_value"
        - hyperparamters_dictionary: the dictionary to add the hyperparameter to.
    returns:
        the dictionary after adding that hyperparameter
    """
    # split the string
    split_hyperparameter_string = hyperparameter_string.split(":")
    hyperparameter_name = split_hyperparameter_string[0]
    hyperparameter_value = split_hyperparameter_string[1].strip()
    hyperparamters_dictionary[hyperparameter_name] = hyperparameter_value
    return "success"

def get_write_string(dictionary, count, type="cbow"):
    to_write = ""
    to_write += "#### COMBINATION " + str(count) + " ####\n"

    # iterate through all the keys in this dictionary
    # and add them along with their values to the values to be written to file
    for key, value in dictionary.items():
        to_write += key + ": " + value + "\n"

    # append the following to the arguments:
    #   - type: unsupervised
    #   - input: dataset/train_500000.txt
    #   - output: word_vectors/cbow_combination_<combo_number>
    to_write += "type: unsupervised\n"
    to_write += "input: " + SCRATCH_VECTORIZE + "dataset/train_500000.txt\n"
    to_write += "output: " + SCRATCH_VECTORIZE + "word_vectors/" + type + "_combination_" + str(count) + "\n"

    # add another newline to differentiate between combinations
    to_write += "\n"
    return to_write

if __name__ == "__main__":
    UNSUPERVISED_HYPERPARAMETERS = []
    UNSUPERVISED_HYPERPARAMETERS.extend([MODEL_TYPE, MINCOUNT_UNSUPERVISED, DIM, MINN_MAXNN, EPOCH_LR_UNSUPERVISED])
    print("unupervised hyperparamters master list:", UNSUPERVISED_HYPERPARAMETERS)
    combinations = get_combinations(UNSUPERVISED_HYPERPARAMETERS)
    print("\n")

    # create a dictionary of args corresponding to each combination of hyperparamters
    # { model_type: skipgram
    #   input_target: dataset/train_1000.txt
    #   minCount: 5
    #   dim: 128
    #   minn: 3
    #   maxn: 6
    #   epoch: 5
    #   lr: 0.05
    #   output_target: word_vectors/train_1000 }

    dictionaries = []
    for i in range(len(combinations)):
        combination = combinations[i]
        # print(combination)
        hyperparamters_dictionary = {}
        for element in combination:
            # element is either a string (hyperparameter) or a tuple (combination of two hyperparamters, whose values are linked)
            if type(element) == str:
                add_hyperparameter(element, hyperparamters_dictionary)
                # # split the string
                # split_element = element.split(":")
                # hyperparameter_name = split_element[0]
                # hyperparameter_value = split_element[1].strip()
                # hyperparamters_dictionary[hyperparameter_name] = hyperparameter_value
            if type(element) == tuple:
                # split each element in tuple
                # add that hyperparameter to hyperparamters_dictionary
                for hyperparameter_string in element:
                    add_hyperparameter(hyperparameter_string, hyperparamters_dictionary)

        print(hyperparamters_dictionary)
        dictionaries.append(hyperparamters_dictionary)
    print("\n")
    print("number of combinations: ", len(dictionaries))

    # write all cbow combinations into a single txt file
    # and all skipgram combinations into a single txt file
    to_write_cbow = ""
    to_write_skipgram = ""
    cbow_count = 0
    skipgram_count = 0
    for dictionary in dictionaries:
        if dictionary['model_type'] == "cbow":
            cbow_count += 1
            to_write_cbow += get_write_string(dictionary, cbow_count, type="cbow")

        elif dictionary['model_type'] == "skipgram":
            skipgram_count += 1
            to_write_skipgram += get_write_string(dictionary, skipgram_count, type="skipgram")

    with open(CBOW_MASTER_SETTINGS_FILE, "w+") as file:
        file.write(to_write_cbow)

    with open(SKIPGRAM_MASTER_SETTINGS_FILE, "w+") as file:
        file.write(to_write_skipgram)
