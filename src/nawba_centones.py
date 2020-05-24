import csv
import json


def load_and_parse_centones_mapping(mapping_path):
    """
    Load csv of centos mapping to dict
    """
    with open(mapping_path, mode="r") as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]: rows[1:] for rows in reader}

    for key in mydict:
        mydict[key] = [x for x in mydict[key] if x != ""]
    return mydict


def load_and_parse_nawba_tabs(path):
    """
    Load and parse nawba to tab mapping
    """
    with open(path, "r") as fp:
        nawba_tabs = json.load(fp)

    tabs_nawba = {}
    for k, v in nawba_tabs.items():
        for t in v:
            tabs_nawba[t] = k.replace("Nawba_", "")
    return tabs_nawba


# def list_scores(data_path):
#     andalusian_description = pd.read_json(os.path.join(data_path, 'andalusian_description.json'))
#     scores = []
#     # Some scores do not have the relevant metadata
#     for i, row in andalusian_description.iterrows():
#         try:
#             tn = row['sections'][0]['tab']['transliterated_name']
#         except:
#             tn = ''
#         scores.append([row['mbid'], tn])
#
#     bad_mbid = []
#
#     for s in scores: # Download scores from dunya through its mbid
#         mbid = s[0]
#         # Some scores aren't available
#         try:
#             score_xml = dunya.docserver.file_for_document(mbid, 'symbtrxml')
#         except:
#             bad_mbid.append(mbid)
#     scores = [x for x in scores if x[0] not in bad_mbid]
#
#     return scores
