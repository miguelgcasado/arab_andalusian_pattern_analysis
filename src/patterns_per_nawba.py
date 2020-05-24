import os
from collections import Counter
import csv
import json
from music21 import *


def mbids_per_tab(andalusian_description):
    """
	Function that creates a dictionary of tubu with the corresponding mbids
	:return: tab_mbid_lookup
	"""
    tab_mbid_lookup = {}
    for i, row in andalusian_description.iterrows():
        try:
            tn = row["sections"][0]["tab"]["transliterated_name"]
        except:
            tn = ""

        tab_mbid_lookup[row["mbid"]] = tn

    return tab_mbid_lookup


def get_tab_from_mbid(mbid, tab_mbid_lookup):
    """
	Function that looks for the matching nawba from a specific mbid
	:param mbid: required mbid
	:param tab_mbid_lookup: lookup dictionary
	:return: corresponding nawba of the required mbid
	"""
    for nawba in tab_mbid_lookup:
        if mbid in tab_mbid_lookup[nawba]:
            found_nawba = nawba
    return found_nawba


def get_repeated_patterns_and_conv_to_wav(nawba_pattern_lookup):
    """
	Function that cleans the nawba_pattern dictionary to keep the
	10 first most repeated patterns in a nawba and save them as wav file
	"""
    for nawba in nawba_pattern_lookup:
        output_folder = os.path.join("../data/patterns_midi", nawba)
        if not os.path.isdir(output_folder):
            os.mkdir(os.path.join(output_folder))

        count = list(Counter([x[0] for x in nawba_pattern_lookup[nawba]]))
        count.sort(key=lambda x: x[1], reverse=True)
        most_relevant_patterns = [x for x in count[:10]]

        for pattern in most_relevant_patterns:
            for p in nawba_pattern_lookup[nawba]:
                if pattern == p[0]:
                    conv_stream_to_midi(p, output_folder)

        nawba_pattern_lookup[nawba] = most_relevant_patterns

    return nawba_pattern_lookup


def conv_stream_to_midi(string_stream, output_folder):
    """
	Function that converts a music21 stream to MIDI file.
	:param string_stream: music21 stream input
	:param output_folder: output folder to stored MIDI file
	"""
    midi_file = output_folder + "/" + string_stream[0] + ".mid"
    string_stream[1].write("midi", midi_file)


def get_m21_patterns(json_path, mbid):
    """
	Function that converts the SIA patterns from the output json in music21 streams.
	:param json_path: Pth for the required file.
	:param mbid: Mbid relatedto the file
	:return: Dictionary containing the output music21 streams.
	"""
    with open(json_path) as json_file:
        results = json.load(json_file)

    m21patterns = {}
    for attempt in results:
        try:
            sia_patterns = [p for p in results[attempt] if p[-1][0] - p[0][0] < 25000]
            sia_patterns = order_sia_patterns(sia_patterns)
            m21patterns[attempt] = conv_SIA_pattern_to_m21pattern(mbid, sia_patterns)
        except Exception as e:
            print("{} contains chords and wont be counted".format(mbid))
    return m21patterns


def conv_SIA_pattern_to_m21pattern(mbid, pattern_list):
    """
	Function that, given a mbid, gives you the SIA patterns in music21 stream format in a list of lists
	:param mbid: required mbid
	:param pattern_list: list of patterns from SIA algorithm
	:return:
	"""
    fn = os.path.join("../data/scores_xml/", mbid + ".xml")
    s = converter.parse(fn)
    p = s.parts[0]
    notes = p.flat.notesAndRests.stream()

    m21_patterns = {}
    for cont in pattern_list:
        patterns_m21 = []
        for pattern in pattern_list[cont]:
            first_note_midi = pattern[0][0]
            last_note_midi = pattern[-1][0]
            first_note_xml = None
            last_note_xml = None
            for note in notes:
                if midi.translate.offsetToMidi(note.offset) == first_note_midi:
                    first_note_xml = note.offset
                if midi.translate.offsetToMidi(note.offset) == last_note_midi:
                    last_note_xml = note.offset
            seg = p.getElementsByOffset(
                first_note_xml,
                last_note_xml,
                mustBeginInSpan=False,
                includeElementsThatEndAtStart=False,
            ).stream()
            patterns_m21.append(seg)
        m21_patterns[cont] = patterns_m21

    return m21_patterns


def conv_m21pattern_to_namenote(pattern):
    """
	Function that transforms a m21 stream in a list with names of the events
	:param pattern: m21 stream
	:return: list of the names of the events
	"""
    midipattern = [
        note.name if note.isNote else "R"
        for note in pattern.flat.notesAndRests.stream()
    ]

    return midipattern


def represent_m21patterns(pattern):
    """
	Function that transforms a m21 stream in a list with names of the events
	:param pattern: m21 stream
	:return: list of the names of the events
	"""
    midipattern = [note.name for note in pattern.flat.notesAndRests.stream()]

    return midipattern


def score_annotations_lookup():
    """
	Function that returns a dictionary with the sections of each mbid.
	:return:
	"""
    with open("../data/arab_andalusian_scores_sections_rafa.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")

        score_annotations = {}
        for row in readCSV:
            if row[0] != "mbid":
                if row[0] not in score_annotations:
                    score_annotations[row[0]] = {}
                if row[1] not in score_annotations[row[0]]:
                    score_annotations[row[0]][row[1]] = [[row[2], row[3]]]
                else:
                    score_annotations[row[0]][row[1]].append([row[2], row[3]])

    return score_annotations


def which_key(midi_pattern, order_sia_pattern):
    """
	Function that gives the index which contains a specific midi_pattern
	:param midi_pattern: pattern to search
	:param order_sia_pattern: dictionary to be searched
	:return: index
	"""
    to_return = 0
    for cont in order_sia_pattern:
        pattern = [x[1] for x in order_sia_pattern[cont][0]]
        if midi_pattern == pattern:
            to_return = cont
    return to_return


def order_sia_patterns(sia_patterns):
    """
	Functions that order the link the sia patterns with the one that contains the same notes
	:param sia_patterns: sia_pattern list of list
	:return: order sia_pattern list of dicts
	"""
    cont = 0
    cont_sia = 0
    order_sia_patterns = {}
    for pattern in sia_patterns:
        midi_pattern = [x[1] for x in pattern]
        if order_sia_patterns:
            cont_sia = which_key(midi_pattern, order_sia_patterns)
        if cont_sia == 0:
            cont += 1
            order_sia_patterns[cont] = [pattern]
        else:
            order_sia_patterns[cont_sia].append(pattern)

    return order_sia_patterns


def mbids_per_nawba(mbid_tab_lookup, nawba_tabs):
    """
	Function that computes the mbid nawba lookup.
	:param mbid_tab_lookup: mbid tab lookup
	:param nawba_tabs: dictionary map of tubus and nawbat
	:return:
	"""
    mbid_nawba_lookup = {}
    for mbid in mbid_tab_lookup:
        if mbid_tab_lookup[mbid] in nawba_tabs:
            nawba = nawba_tabs[mbid_tab_lookup[mbid]]
            mbid_nawba_lookup[mbid] = nawba
    return mbid_nawba_lookup


def conv_pattern_list_to_string(nawba_patterns_lookup):
    """
	Funcion that transform list of list of patterns in list of string of patterns.
	:param nawba_patterns_lookup: nawba patterns found dictionary
	:return: nawba patterns found dictionary with patterns=strings.
	"""
    nawba_patterns_lookup2 = {}
    for nawba in nawba_patterns_lookup:
        for pattern in nawba_patterns_lookup[nawba]:
            if nawba not in nawba_patterns_lookup2:
                nawba_patterns_lookup2[nawba] = [["".join(pattern[0]), pattern[1]]]
            else:
                nawba_patterns_lookup2[nawba].append(["".join(pattern[0]), pattern[1]])
    return nawba_patterns_lookup2


def parse_SIA_patterns(path, mbid_nawba_lookup):
    """
	Function that goes over SIA output in a given path and generates a list of output patterns per nawba with length
	(3 - 10)
	:param path: path containing the output JSON files from SIA
	:param mbid_nawba_lookup: lookup dictionary linking mbid and corresponding nawba
	:return: lookup dictionary containing linking nawba and repeated patterns on that nawba.
	"""
    nawba_patterns_lookup = {}
    for mbid in mbid_nawba_lookup:
        file = mbid + ".json"
        nawba = mbid_nawba_lookup[mbid]
        print(file + "...")
        m21patterns = None
        with open(os.path.join(path, file)) as json_file:
            results = json.load(json_file)
        try:
            sia_patterns = [
                p for p in results["SiaTonic4"] if p[-1][0] - p[0][0] < 100000
            ]
            sia_patterns = order_sia_patterns(sia_patterns)
            m21patterns = conv_SIA_pattern_to_m21pattern(mbid, sia_patterns)
        except Exception as e:
            print("{} contains chords and wont be counted".format(mbid))
        if m21patterns != None:
            for idx in m21patterns:
                for seg in m21patterns[idx]:
                    midipatternset = conv_m21pattern_to_namenote(seg)
                    pattern_no_silences = [x for x in midipatternset if x != "R"]
                    if len(pattern_no_silences) >= 3 and len(pattern_no_silences) <= 10:
                        if nawba not in nawba_patterns_lookup:
                            nawba_patterns_lookup[nawba] = [[midipatternset, seg]]
                        else:
                            nawba_patterns_lookup[nawba].append([midipatternset, seg])

    return nawba_patterns_lookup
