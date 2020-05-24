import os
import json
import numpy as np
import csv
from music21 import *
from compmusic import dunya
import matplotlib.pyplot as plt

dunya.set_token("52fc6ac49c0b7fc9644404aaf4f9bc1a7088d69d")

# Get the results in an array
with open(
    "/home/miguelgc96/Desktop/a451a7fc-c53f-462a-b3fc-4377bb588105.json"
) as json_file:
    data = json.load(json_file)

fn = "data/scores_xml/a451a7fc-c53f-462a-b3fc-4377bb588105.xml"
s = converter.parse(fn)
p = s.parts[0]
notes = p.flat.notesAndRests.stream()
with open("../sections_scores_annotations/arab_andalusian_sections.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")

    score_annotations = {}
    for row in readCSV:
        if row[0] != "mbid":
            if row[0] not in score_annotations:
                score_annotations[row[0]] = {row[1]: row[2]}
            else:
                score_annotations[row[0]][row[1]] = row[2]

    mbid = "a451a7fc-c53f-462a-b3fc-4377bb588105"
    last_midi_offset = midi.translate.offsetToMidi(notes[-1].offset)

cont = 0
parameters = [
    [0.95, 0],
    [0.4, 0],
    [0.7, 0],
    [0.6, 0],
    [0.8, 0],
    [0.75, 0],
    [0.9, 0],
    [0.85, 0],
]

for label in data:
    patterns = data[label]
    print(
        "(minCompactness = "
        + str(parameters[cont][0])
        + " , minDensity = "
        + str(parameters[cont][1])
        + ") =>{} patterns were found".format(len(patterns))
    )

    # FILTER EVENTS TO GET SUBSET OF PATTERNS WITH SOME CHARACTERISTICS:

    # Get patterns with less than 21 notes:
    # minipatterns_21notes = [p for p in patterns if len(p)<21]
    # print('\n{} patterns with less than 21 events were found'.format(len(minipatterns_21notes)))

    # Get patterns with less than 100000 between first and last note of the pattern:
    # minipatterns_offset = [p for p in patterns if p[-1][0]-p[0][0] < 100000]
    # print('\n{} patterns with with narrow offset were found'.format(len(minipatterns_offset)))

    # Get patterns with correlative notes:

    # Get patterns with different midi values

    # # UNDERSTAND THE MIDI OFFSET AND THE CONVERSION TO THE SCORE.
    # patterns_offset = []
    # for pattern in patterns:
    # 	first_note_midi = pattern[0][0]
    # 	last_note_midi = pattern[-1][0]
    # 	first_note_xml = None
    # 	last_note_xml = None
    # 	for note in notes:
    # 		if midi.translate.offsetToMidi(note.offset) == first_note_midi:
    # 			first_note_xml = note.offset
    # 		if midi.translate.offsetToMidi(note.offset) == last_note_midi:
    # 			last_note_xml = note.offset
    # 	if first_note_xml != None and last_note_xml != None:
    # 		patterns_offset.append({'first note': first_note_xml, 'last note': last_note_xml})

    # Represent the different discovered patterns in the score editor:
    # for pattern in patterns_offset[:5]:
    # 	seg = p.getElementsByOffset(pattern['first note'], pattern['last note'],
    # 								mustBeginInSpan=False,
    # 								includeElementsThatEndAtStart=False).stream()
    # 	seg.show()

    # Represent minipatterns_offset in the timeline of the score with annotation of the sections:
    plt.figure()
    for section in score_annotations[mbid]:
        section_offset = midi.translate.offsetToMidi(
            float(score_annotations[mbid][section])
        )
        if section_offset < last_midi_offset:
            plt.vlines(section_offset, 55, 75, colors=np.random.rand(3,), label=section)

    for p in patterns:
        midi_offset = [x[0] for x in p]
        midi_values = [x[1] for x in p]

        plt.plot(midi_offset, midi_values, "-o", c=np.random.rand(3,))
    plt.legend()
    plt.xlabel("Score offset")
    plt.ylabel("MIDI values")
    plt.title(
        "minCompactness = "
        + str(parameters[cont][0])
        + " , minDensity = "
        + str(parameters[cont][1])
    )
    cont += 1
plt.show()
