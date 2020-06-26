import os
import music21
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


def pattern_stream_from_score(path, rest_quarter_length=0):
    """
    Load a score from <path> and return an ordered list of notes
    R represents a rest greater than or equal to <rest_quarter_length>
    ...rests shorter than <rest_quarter_length> are ignored

    Fails if score contains chords
    """

    s = music21.converter.parse(path)
    p = s.parts[0]

    # These are all the notes of the whole piece, fails for chords
    notes_and_rests = p.flat.notesAndRests.stream()
    notes = []
    for n in notes_and_rests:
        if n.isRest:
            notes.append("R")
        else:
            notes.append(n.name)
    return notes


def extract_pattern_grams(notes, min_n=2, max_n=2):
    """
    For a list of list of notes, <notes>
    Extract all possible note-grams up to a maximum length of <n>
    Converts stream of notes to bag-of-patterns
    """
    num_notes = len(notes)
    comb = []
    for i in range(num_notes):
        # Final n patterns are counted more than once
        n_ = num_notes - i if max_n > num_notes - i else max_n
        comb.append([notes[i : i + j] for j in range(2, n_ + 1)])
    flat = [i for c in comb for i in c]
    return [x for x in flat if len(x) >= min_n]


def get_notes_from_score(scores_path, mbid_nawba_lookup):
    """
    Function that return a dict with mbid: all notes of a score
    :param scores_path: path where the scores are saved
    :return: described dict
    """
    notes_dict = {}
    chord_mbid = []
    for root, dirs, files in os.walk(scores_path):
        for file in files:
            mbid = file.replace(".xml", "")
            if mbid in mbid_nawba_lookup:
                # Fails for scores with chords
                try:
                    note_stream = pattern_stream_from_score(
                        os.path.join(scores_path, file)
                    )
                    notes_dict[mbid] = "".join(note_stream)
                except Exception as e:
                    print("{} contains chords and wont be counted".format(mbid))
                    chord_mbid.append(mbid)
    return notes_dict


def get_bag_of_patterns(notes_dict):
    mbid_patterns_dict = {}
    for mbid in notes_dict:
        all_patterns = [extract_pattern_grams(notes_dict[mbid], min_n=3, max_n=10)]
        mbid_patterns_dict[mbid] = [
            item for sublist in all_patterns for item in sublist
        ]
    return mbid_patterns_dict


def get_pattern_ocurrence(nawba_patterns_lookup, tab_mbid_lookup):
    """
    For a list of all recordings patterns
    Returns:
        list, each elecment a recording, summarised as a list of (pattern, total_count_of_pattern)
    """
    mbid_pattern_occurences = {}
    for mbid in tab_mbid_lookup:
        score_path = "../data/scores_xml/" + mbid + ".xml"
        score_string = "".join(pattern_stream_from_score(score_path))
        for pattern in nawba_patterns_lookup.values():
            occurrences = score_string.count(pattern)
            if mbid not in mbid_pattern_occurences:
                mbid_pattern_occurences[mbid] = [score_string.count(occurrences)]
            else:
                mbid_pattern_occurences[mbid].append(occurrences)

    return mbid_pattern_occurences


def train_classifier(data, mbid_nawba_lookup, plotMatrix=False):
    """
    Function that train and test classifier for our data.
    :param data: structured data using number of occurrences of each pattern as feature
    :param mbid_nawba_lookup: dict lookup linking mbid and nawba
    :param plotMatrix: boolean to plot Confusion Matrix of the classifier
    :return: accuracy of the classifier
    """

    train, test = train_test_split(data, test_size=0.6)

    y = [mbid_nawba_lookup[x[0]] for x in train]
    X = [
        x[1] for x in train
    ]  # array containing the number of occurrences of each found_pattern
    clf = LogisticRegression(
        random_state=42, C=0.01, solver="liblinear", multi_class="ovr"
    ).fit(X, y)

    to_predict = [x[1] for x in test]
    y_pred = clf.predict(to_predict)
    y_true = [mbid_nawba_lookup[x[0]] for x in test]

    if plotMatrix:
        plt.figure(figsize=(10, 10))
        plt.title('Confusion matrix of nawba classifier using pattern occurrences')
        sn.heatmap(confusion_matrix(y_pred, y_true), linewidths=1, annot=True)
        plt.xlabel('Predicted Nawba')
        plt.ylabel('True Nawba')
        plt.show()

    return accuracy_score(y_pred, y_true)
