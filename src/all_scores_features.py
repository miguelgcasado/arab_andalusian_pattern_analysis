import music21 as m21
import glob
import json
import symbolic_features
import patterns_per_nawba as pn
import nawba_centones
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

andalusian_description = pd.read_json("../data/andalusian_description.json")
mbid_tab_lookup = pn.mbids_per_tab(andalusian_description)
nawba_tabs = nawba_centones.load_and_parse_nawba_tabs("../data/nawba_tabs.json")
mbid_nawba_lookup = pn.mbids_per_nawba(mbid_tab_lookup, nawba_tabs)
midi_files = [
    f
    for f in glob.glob("../results/patterns_midi/sia/**/*.mid", recursive=True)
    if "hidefornow" not in f
]


def separate_string_pattern_in_notes(pattern):
    """
    Function that separate a string of notes in list of notes
    """
    output = []
    cont = 0
    for idx in range(len(pattern) - 1):
        if pattern[idx + 1] == "#" or pattern[idx + 1] == "-":
            output.append(pattern[idx] + pattern[idx + 1])
        elif pattern[idx] != "#" and pattern[idx] != "-":
            output.append(pattern[idx])
    if pattern[-1] != "#" and pattern[-1] != "-":
        output.append(pattern[-1])
    return output


def get_output_SIA_features():
    """
    Function that returns dictionary with pattern as corresponding symbolic features
    """
    pattern_features = []
    for file in midi_files:
        pattern_name = file.split("/")[-1].replace(".mid", "")
        # print("processing " + pattern_name + '...')
        pattern_features.append(
            [pattern_name, symbolic_features.get_features_array(file)]
        )

    midi_pattern_features = {}
    for row in pattern_features:
        midi_pattern_features[row[0]] = row[1]

    return midi_pattern_features


def compute_mbid_pattern_occurrence():
    """
    Function that computes a dict with {mbid: SIA patterns in score} and store in json.
    """
    xml_files = [f for f in glob.glob("../data/scores_xml/*.xml", recursive=True)]
    with open("../results/nawba_patterns_lookup_post_clean_095.json") as json_file:
        nawba_patterns_lookup = json.load(json_file)

    pattern_list = [x for y in nawba_patterns_lookup.values() for x in y]
    mbid_pattern_occurrence_dict = {}
    cont = 1
    for xml_score in xml_files:
        print("({}/{})...".format(cont, len(xml_files)))
        mbid = xml_score.split("/")[-1].replace(".xml", "")
        if mbid != "2d8e2820-e4cf-4dc8-b4f1-45f8fb65de9e":  # contains chords
            s = m21.converter.parse(xml_score)

            p = s.parts[0]
            notes_and_rests = p.flat.notesAndRests.stream()
            for pattern in pattern_list:
                length = len(separate_string_pattern_in_notes(pattern))
                for i in range(len(notes_and_rests[: -length + 1])):
                    buffer = []
                    for j in range(length):
                        buffer.append(notes_and_rests[i + j])
                    phrase = ""
                    for n in buffer:
                        if n.isRest:
                            phrase += "R"
                        else:
                            phrase += n.name
                    if phrase == pattern:
                        if mbid not in mbid_pattern_occurrence_dict:
                            mbid_pattern_occurrence_dict[mbid] = pattern_features[
                                pattern
                            ]
                        else:
                            mbid_pattern_occurrence_dict[mbid].append(
                                pattern_features[pattern]
                            )

        mbid_pattern_occurrence_dict[mbid] = np.mean(
            mbid_pattern_occurrence_dict[mbid], axis=0
        )
        cont += 1

    with open("../results/result_score_features.json", "w") as outfile:
        json.dump(mbid_pattern_occurrence_dict, outfile)


def compute_X_Y_model(
    mbid_pattern_occurrence_dict, midi_pattern_features, statistic="median"
):
    """
    Function that computes X,Y matrix to feed nawba classifier model.
    :param mbid_pattern_occurrence_dict
    :param midi_pattern_features
    :param statistic: (median, mean, max, min)
    :return:
    """
    pattern_names = [file.split("/")[-1].replace(".mid", "") for file in midi_files]
    X = {}
    for mbid in mbid_nawba_lookup:
        if mbid in mbid_pattern_occurrence_dict:
            for pattern in mbid_pattern_occurrence_dict[mbid]:
                if pattern[0] in pattern_names:
                    if mbid not in X:
                        # print(pattern)
                        X[mbid] = [midi_pattern_features[pattern[0]]]
                    else:
                        X[mbid].append(midi_pattern_features[pattern[0]])
            if statistic == "median":
                X[mbid] = np.median(X[mbid], axis=0)
            if statistic == "mean":
                X[mbid] = np.mean(X[mbid], axis=0)
            if statistic == "max":
                X[mbid] = np.max(X[mbid], axis=0)
            if statistic == "min":
                X[mbid] = np.min(X[mbid], axis=0)

    Y = [mbid_nawba_lookup[mbid] for mbid in X]

    return X, Y


def train_classifier_features(X, Y, plotMatrix=False):
    """
    Function that computes nawba classifier model and return prediction accuracy and improtance of the features.
    """
    data = []
    cont = 0
    for mbid in X:
        data.append([X[mbid], Y[cont]])
        cont += 1

    train, test = train_test_split(data, test_size=0.6)

    Y_train = [x[1] for x in train]
    X_train = [
        x[0] for x in train
    ]  # array containing the number of occurrences of each found_pattern
    clf = LogisticRegression(
        random_state=42, C=0.01, solver="liblinear", multi_class="ovr"
    ).fit(X_train, Y_train)
    importance = clf.coef_[0]

    X_test = [x[0] for x in test]
    Y_pred = clf.predict(X_test)
    Y_test = [x[1] for x in test]

    if plotMatrix:
        plt.figure(figsize=(10, 10))
        plt.title('Confusion matrix of nawba classifier using pattern occurrences and features')
        sn.heatmap(confusion_matrix(Y_pred, Y_test),  linewidths=1, annot=True)
        plt.xlabel('Predicted Nawba')
        plt.ylabel('True Nawba')
        plt.show()

    return accuracy_score(Y_pred, Y_test), importance
