import os
import json
from os import listdir, makedirs
from os.path import isfile, join, splitext, exists

# Assume the data set is in the below subfolder
inputDataPrefix = "./data/"

# Loads the samples in the train, validation, or test set
def loadMOROCODataSamples(subsetName):
# Copyright for function (C) 2018  Andrei M. Butnaru, Radu Tudor Ionescu
    inputSamplesFilePath = (inputDataPrefix + "%s/samples.txt") % (subsetName)
    inputDialectLabelsFilePath = (inputDataPrefix + "%s/dialect_labels.txt") % (subsetName)
    inputCategoryLabelsFilePath = (inputDataPrefix + "%s/category_labels.txt") % (subsetName)
    
    IDs = []
    samples = []
    dialectLabels = []
    categoryLabels = []
    
    # Loading the data samples
    inputSamplesFile = open(inputSamplesFilePath, 'r')
    sampleRows = inputSamplesFile.readlines()
    inputSamplesFile.close()

    for row in sampleRows:
        components = row.split("\t")
        IDs += [components[0]]
        samples += [" ".join(components[1:])]

    # Loading the dialect labels
    inputDialectLabelsFile = open(inputDialectLabelsFilePath, 'r')
    dialectRows = inputDialectLabelsFile.readlines()
    inputDialectLabelsFile.close()
    
    for row in dialectRows:
        components = row.split("\t")
        dialectLabels += [int(components[1])]
    
    # Loading the category labels
    inputCategoryLabelsFile = open(inputCategoryLabelsFilePath, 'r')
    categoryRows = inputCategoryLabelsFile.readlines()
    inputCategoryLabelsFile.close()
    
    for row in categoryRows:
        components = row.split("\t")
        categoryLabels += [int(components[1])]

    # IDs[i] is the ID of the sample samples[i] with the dialect label dialectLabels[i] and the category label categoryLabels[i]
    return IDs, samples, dialectLabels, categoryLabels

def build_instruction_set(task_ids, task_samples, task_labels, format="mistral", task="dialect"):
    """
    Build an instruction set for a specified task in a given format.

    Parameters:
    - task_ids (list): ids from MOROCO
    - task_samples (list): text samples
    - task_labels (list): labels for the given task
    - format (str, optional): model to be used, for the moment Mistral
    - task (str, optional): unused, maybe to switch to other Vardial tasks

    Returns:
    - instruction_set (str): json set with raw and instruction texts
    """

    json_set = []
    for id, sample, label in zip(task_ids, task_samples, task_labels):
        instruction = f"[INST] O să primești un fragment dintr-un articol de știri scris în limba română. Trebuie să îl clasifici în dialectul standard al limbii române, sau în dialectul moldovenesc, folosit în Republica Moldova. Numele de persoane sau de locuri geografice au fost schimbate în \"$NE#\", ca să fie împiedicată folosirea de denumiri specifice pentru identificare, în loc de proprietăți lingvistice.\nFragmentul este acesta:\"{sample}\"\n Alege unul dintre cele doua dialecte pentru clasificare:\n1. dialectul moldovenesc\n2. dialectul standard\n[/INST] Dialectul din fragment este {label}."

        json_set.append({
            'id': id,
            'raw_sample': sample,
            'instr_sample': instruction,
            'dialect': label
        })
    
    return json_set

def write_set(split, out_root, format="mistral", task="dialect"):
    task_ids, task_samples, task_dialect, task_category = loadMOROCODataSamples(split)
    task0_set = build_instruction_set(task_ids, task_samples, task_dialect, format=format, task=task)
    task0_fp =  os.path.join(out_root, f'{split}_model={format}_task={task}.jsonl')
    with open(task0_fp, 'w') as f:
        for obj in task0_set:
            json.dump(obj, f)
            f.write('\n')

if __name__ == "__main__":
    write_set("train", "data/")