# Preprocess the data:
#       - assign the final labels based on confidence threshold.
#       - TODO: [potentially] remove non-English projects
#       - lump "Slideshow" projects into "Other" category [optional]
#
#       - textify projects in the dataset file (using scratch-textify)
#       - convert the dataset file into supervised format: i.e. add __label__
#       - shuffle the dataset and get train and test splits
#       - write out final version to a text file.

# NOTE: scratch-textify might end up diminishing our project list even further,
#       since some projects might've gotten deleted or otherwise not available on Scratch anymore.

import os
import pandas as pd
import subprocess

CONFIDENCE_THRESHOLD = 0.7
DATASET = "./dataset"
DATASET_TARGET = os.path.abspath(os.path.join(DATASET, 'dataset.csv'))
PROJECTS_TARGET = os.path.abspath(os.path.join(DATASET, 'project_ids.csv'))
TEXTIFY = "../../scratch-textify"
TEXTIFY_TARGET = os.path.abspath(os.path.join(TEXTIFY, 'index-pool.js'))
BATCH_SIZE = 100

def assign_final_labels(projects, threshold=CONFIDENCE_THRESHOLD):
    """
    For each project, we will assign the final label based on the confidence level:
        - if confidence >= threshold: then the label is taken as is.
        - if confidence < threshold: then the label is changed to Other, and the confidence is set to 1.

    inputs:
        - threshold: confidence threshold above which that category is retained
                    (and below which it gets transformed into "Other")
        - projects: projects dataframe.

    returns:
        - new projects dataframe
    """

    def replace_label(row):
        if row['confidence'] < threshold:
            row['label'] = "Other"
            row['confidence'] = 1.0 # change the confidence to 1 that it's Other
        return row

    projects = projects.apply(replace_label, axis=1, result_type='broadcast')
    projects = pd.DataFrame(projects)

    return projects

def lump_categories(projects, from_label="Slideshow", to_label="Other"):
    """
    Lumps the projects in the "from_label" category into projects in the "to_label" category.

    inputs:
        - projects: projects dataframe.
        - from_label: "from" category.
        - to_label: "to" category.

    returns:
        - new projects dataframe
    """
    def replace_label(row):
        if row['label'] == from_label:
            row['label'] = to_label
        return row

    projects = projects.apply(replace_label, axis=1, result_type='broadcast')
    projects = pd.DataFrame(projects)
    return projects

if __name__ == "__main__":
    # read the dataset into dataframe
    df = pd.read_csv(DATASET_TARGET)

    # assign the final labels based on confidence threshold
    df = assign_final_labels(df, threshold=CONFIDENCE_THRESHOLD)

    # lump "Slideshow" projects into "Other" category.
    df = lump_categories(df, from_label="Slideshow", to_label="Other")

    # write out the project ids for the classification task to a csv file (useful to have)
    project_ids = df['id'].tolist()

    with open(PROJECTS_TARGET, 'w+') as file:
        [file.write((str(project_id) + "\n")) for project_id in project_ids]

    # # textify projects in the dataset file (using scratch-textify)
    # dataset_directory = os.path.abspath(DATASET)
    #
    # command = ["node", TEXTIFY_TARGET, "--projects_file", PROJECTS_TARGET, "--batch_size", str(BATCH_SIZE), "--write_target", dataset_directory, "--textify_directory", TEXTIFY]
    # print("command: ", command)
    # print("\n")
    # code = subprocess.check_call(command)

    # combine all the .ids files together, all the .txt files together, all the .err files together


    # convert the dataset into supervised format: i.e. add __label__

    # shuffle the dataset and get train and test splits

    # write out final version to a text file.
