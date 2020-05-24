import os
from music21 import *
import json
import nawba_centones

data_path = "../data"
nawba_centones_lookup = nawba_centones.load_and_parse_centones_mapping(
    os.path.join(data_path, "centones_nawba.csv")
)


def separate_string_pattern_in_notes(pattern):
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


with open("../results/patterns_midi/tfidf/tfidf_patterns.json") as json_file:
    tfidf_patterns = json.load(json_file)

for nawba in tfidf_patterns:
    for cento in tfidf_patterns[nawba]:
        s = stream.Stream()
        for n in separate_string_pattern_in_notes(cento):
            s.append(note.Note(n, type="quarter"))
        s.insert(0, instrument.Sitar())
        s.write(
            "midi", "../results/patterns_midi/tfidf/" + nawba + "/" + cento + ".mid"
        )
