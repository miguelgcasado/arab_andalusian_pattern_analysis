import os
import glob
from music21 import *

files = [f for f in glob.glob("scores_xml/" + "**/*.xml", recursive=True)]

for file in files:
    mbid = file.replace("scores_xml/", "")
    mbid = mbid.replace(".xml", "")
    s = converter.parse(file)
    mf = midi.translate.streamToMidiFile(s)
    mf.open("scores_midi/" + mbid + ".mid", "wb")
    mf.write()
    mf.close()
