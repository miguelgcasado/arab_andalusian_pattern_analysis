import music21 as m21
import numpy as np
import os
import glob
import operator
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt
import json

########################################### RESTS ###########################################################
def average_rest_length(notes_rests):
    av_rest_dur = 0
    nrests = 0
    for note in notes_rests:
        if note.isRest:
            av_rest_dur += note.quarterLength
            nrests += 1
    return float(av_rest_dur / nrests)


########################################## NOTES ############################################################
def number_notes(notes_rests):
    notes = [note for note in notes_rests if note.isNote]
    return len(notes)


def average_note_length(notes_rests):
    av_note_dur = 0
    nnotes = 0
    for note in notes_rests:
        if note.isNote:
            av_note_dur += note.quarterLength
            nnotes += 1
    return av_note_dur / nnotes


def most_repeated_note(notes_rests):
    note_dict = {}
    only_notes = [note for note in notes_rests if note.isNote]
    for note in only_notes:
        if note.pitch.midi not in note_dict:
            note_dict[str(note.pitch.midi)] = 1
        else:
            note_dict[str(note.pitch.midi)] += 1
    return float(max(note_dict.items(), key=operator.itemgetter(1))[0])


def average_three_most_repeated_note(notes_rests):
    note_dict = {}
    only_notes = [note for note in notes_rests if note.isNote]
    for note in only_notes:
        if note.pitch.midi not in note_dict:
            note_dict[str(note.pitch.midi)] = 1
        else:
            note_dict[str(note.pitch.midi)] += 1
    return np.mean(max(note_dict.items(), key=operator.itemgetter(1))[:3])


def first_note(notes_rests):
    if notes_rests[0].isNote:
        return notes_rests[0].pitch.midi
    else:
        return 0


def last_note(notes_rests):
    if notes_rests[-1]:
        return notes_rests[-1].pitch.midi
    else:
        return 0


def longest_note(notes_rests):
    note_dict = {}
    for note in notes_rests:
        if note.isNote:
            name = note.pitch.midi
        else:
            name = 0
        if note.name not in note_dict:
            note_dict[name] = note.quarterLength
        else:
            note_dict[name] += note.quarterLength
    return max(note_dict.items(), key=operator.itemgetter(1))[0]


def mean_midi_value(notes_rests):
    midi_values = [note.pitch.midi for note in notes_rests if note.isNote]
    return np.mean(midi_values)


def contain_x(notes_rests):
    check = 0
    for note in notes_rests:
        if "#" in note.name:
            check = 1
    return check


def contain_b(notes_rests):
    check = 0
    for note in notes_rests:
        if "-" in note.name:
            check = 1
    return check


def note_first_pulse_bar(notes_rests):
    return [note.pitch.midi for note in notes_rests if note.beat == 1 and note.isNote]


def note_density_per_quarter_note(notes_rests):
    nnotes = len([note for note in notes_rests])
    dur = 0
    for note in notes_rests:
        dur += note.quarterLength

    return nnotes / dur


def is_continuous(notes_rests):
    check = 1
    only_notes = [note for note in notes_rests if note.isNote]
    for i in range(len(only_notes) - 1):
        if (
            only_notes[i].isNote and only_notes[i + 1].isNote
        ):  # correct with intervals with silences in between
            if (only_notes[i + 1].pitch.midi - only_notes[i].pitch.midi) > 2:
                check = 0
    return check


def contains_dot(notes_rests):
    check = 0
    for note in notes_rests:
        if note.duration.dots:
            check = 1
    return check


def over_first_octave(notes_rests):
    check = 0
    for note in notes_rests:
        if note.isNote:
            if note.pitch.midi >= 72:
                check = 1
    return check


def under_first_octave(notes_rests):
    check = 0
    for note in notes_rests:
        if note.isNote:
            if note.pitch.midi <= 60:
                check = 1
    return check


############################################ linearity ##########################################################
def direction(notes_rests):
    only_notes = [note for note in notes_rests if note.isNote]
    check = 0
    if only_notes[0].pitch.midi < only_notes[-1].pitch.midi:
        check = 1
    elif only_notes[0].pitch.midi > only_notes[-1].pitch.midi:
        check = -1
    elif only_notes[0].pitch.midi == only_notes[-1].pitch.midi:
        check = 0
    return check


############################################# intervals #########################################################
def interval_first_last_note(notes_rests):
    only_notes = [note for note in notes_rests if note.isNote]
    return m21.interval.Interval(
        noteStart=only_notes[0], noteEnd=only_notes[-1]
    ).semitones


def interval_two_last_notes(notes_rests):
    only_notes = [note for note in notes_rests if note.isNote]
    return m21.interval.Interval(
        noteStart=only_notes[-2], noteEnd=only_notes[-1]
    ).semitones


def most_repeated_interval(notes_rests):
    interval_dict = {}
    only_notes = [note for note in notes_rests if note.isNote]
    for i in range(len(only_notes) - 1):
        interval = m21.interval.Interval(
            noteStart=only_notes[i], noteEnd=only_notes[i + 1]
        ).semitones
        if interval not in interval_dict:
            interval_dict[str(interval)] = 1
        else:
            interval_dict[str(interval)] += 1
    return float(max(interval_dict.items(), key=operator.itemgetter(1))[0])


# def interval_interval(notes_rests):


def number_M_intervals(notes_rests):
    cont = 0
    only_notes = [note for note in notes_rests if note.isNote]
    for i in range(len(only_notes) - 1):
        interval = m21.interval.Interval(
            noteStart=only_notes[i], noteEnd=only_notes[i + 1]
        ).name
        if "M" in interval:
            cont += 1
    return cont


def number_m_intervals(notes_rests):
    cont = 0
    only_notes = [note for note in notes_rests if note.isNote]
    for i in range(len(only_notes) - 1):
        interval = m21.interval.Interval(
            noteStart=only_notes[i], noteEnd=only_notes[i + 1]
        ).name
        if "m" in interval:
            cont += 1
    return cont


########################################## rhythm ###################################################################


def range_rhythmic(notes_rests):
    dur = [note.quarterLength for note in notes_rests]
    return float(max(dur) - min(dur))


###################################################################################################


def get_features_array(file):
    """
    Function that computes an array of symboic features for a specific MIDI/musicXML file.
    :param file: input symbolic file
    :return: array of features
    """
    s = m21.converter.parse(file)
    notes_rests = s.flat.notesAndRests
    features_array = []

    features_array.append(average_rest_length(notes_rests))
    features_array.append(number_notes(notes_rests))
    features_array.append(average_note_length(notes_rests))
    features_array.append(most_repeated_note(notes_rests))
    # features_array.append(average_three_most_repeated_note(notes_rests)) correct
    features_array.append(first_note(notes_rests))
    features_array.append(last_note(notes_rests))
    features_array.append(longest_note(notes_rests))
    features_array.append(mean_midi_value(notes_rests))
    features_array.append(contain_b(notes_rests))
    features_array.append(contain_x(notes_rests))
    # features_array.append(note_first_pulse_bar(notes_rests)) will be corrected when using the streams from the notebook instead of midis
    features_array.append(note_density_per_quarter_note(notes_rests))
    features_array.append(is_continuous(notes_rests))
    features_array.append(contains_dot(notes_rests))
    features_array.append(over_first_octave(notes_rests))
    features_array.append(under_first_octave(notes_rests))
    features_array.append(direction(notes_rests))
    features_array.append(interval_first_last_note(notes_rests))
    features_array.append(interval_first_last_note(notes_rests))
    features_array.append(interval_two_last_notes(notes_rests))
    features_array.append(most_repeated_interval(notes_rests))
    features_array.append(number_M_intervals(notes_rests))
    features_array.append(number_m_intervals(notes_rests))
    features_array.append(range_rhythmic(notes_rests))

    return features_array


feature_names = [
    "average_rest_length",
    "number_notes",
    "average_note_length",
    "most_repeated_note",
    "first_note",
    "last_note",
    "longest_note",
    "mean_midi_value",
    "contain_b",
    "contain_x",
    "note_density_per_quarter_note",
    "is_continuous",
    "contains_dot",
    "over_first_octave",
    "under_first_octave",
    "direction",
    "interval_first_last_note",
    "interval_first_last_note",
    "interval_two_last_notes",
    "most_repeated_interval",
    "number_M_intervals",
    "number_m_intervals",
    "range_rhythmic",
]
