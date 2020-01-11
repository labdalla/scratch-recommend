import argparse
import os
import subprocess

FASTTEXT = "../fastText-0.9.1"
FASTTEXT_TARGET = os.path.abspath(os.path.join(FASTTEXT, 'fasttext'))
SETTINGS = "./settings"
SETTINGS_TARGET = os.path.abspath(os.path.join(SETTINGS, 'settings.txt'))


def read_settings(filename):
    settings = {}
    with open(filename) as file:
        # read in the file
        content = file.read()

        # split by newline \n
        lines = content.split("\n")

        # remove the last element of "lines" if it's a ""
        if lines[-1] == "":
            lines = lines[:len(lines)-1]

        # split by colon : and fill in the settings dictionary
        lines = [line.split(":") for line in lines]
        for setting_type, setting in lines:
            setting = setting.strip()
            settings[setting_type] = setting

        return settings


def train(args, type = "unsupervised"):
    """
    inputs:
        - args: a dictionary that represents the command line arguments for the fasttext command.
          in the form:
                    { type: unsupervised
                      model_type: skipgram
                      input_target: dataset/train_1000.txt
                      minCount: 5
                      dim: 128
                      minn: 3
                      maxn: 6
                      epoch: 5
                      lr: 0.05
                      output_target: word_vectors/train_1000 }

    returns: True if training finished, False otherwise.
    """
    commandline_args = []

    # add "supervised" to the command if that's the setting
    if type == "supervised":
        commandline_args.append("supervised")

    # parse dictionary into list of arguments
    for setting_type, setting in args.items():
        if setting_type == "model_type":
            # only append the setting and continue to next iteration
            commandline_args.append(setting)
            continue

        commandline_args.append("-" + setting_type)
        commandline_args.append(setting)

    command = [FASTTEXT_TARGET] + commandline_args

    print("command: ", " ".join(command))
    print("\n")
    code = subprocess.check_call(command)

    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wrapper for fasttext command line utility.")
    parser.add_argument('--settings', type = str, help='settings file for the command line arguments')

    args = parser.parse_args()

    if args.settings:
        settings = args.settings
    else:
        settings = SETTINGS_TARGET

    fasttext_args = read_settings(settings)

    type = fasttext_args.get('type', None)

    # remove type argument from fasttext_args dictionary before passing it into the training functions
    del fasttext_args['type']

    if type == "unsupervised":
        code = train(fasttext_args, type = type)

    elif type == "supervised":
        code = train(fasttext_args, type = type)

    print("return code: ", code)
