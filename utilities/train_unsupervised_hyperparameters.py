from fasttext import *

SETTINGS = "./settings"
CBOW_MASTER_SETTINGS_FILE = os.path.abspath(os.path.join(SETTINGS, 'cbow_master_settings.txt'))
SKIPGRAM_MASTER_SETTINGS_FILE = os.path.abspath(os.path.join(SETTINGS, 'skipgram_master_settings.txt'))

# read in the master settngs text file for each of cbow and skipgram and parse the different args dictionaries
# (with a dictionary corresponding to each combination)
def parse_settings(filename):
    master_settings_list = []
    settings = {}
    with open(filename) as file:
        # read in the file
        content = file.read()

        # split by newline \n
        lines = content.split("\n")

        # remove the last element of "lines" if it's a ""
        if lines[-1] == "":
            lines = lines[:len(lines)-1]

        for line in lines:

            # check for start of a combination
            if "#### COMBINATION" in line:
                # append the previous settings dictionary before overwriting it
                if len(settings) != 0:
                    master_settings_list.append(settings)

                settings = {}
                continue
            else:
                if ":" not in line:
                    continue
                setting_type, setting = line.split(":")
                setting = setting.strip()
                settings[setting_type] = setting

    # append the last combination
    master_settings_list.append(settings)
    return master_settings_list

master_settings_list = parse_settings(CBOW_MASTER_SETTINGS_FILE)
# for dictionary in master_settings_list:
#     print(dictionary)
#
# print("\n")
# print("Number of args dictionaries:", len(master_settings_list))

# iterate through each combination and train the unsupervised model on it
for settings in master_settings_list:
    type = settings.get('type', None)
    del settings['type']

    if type == "unsupervised":
        code = train(settings, type = type)
    print("return code: ", code)
