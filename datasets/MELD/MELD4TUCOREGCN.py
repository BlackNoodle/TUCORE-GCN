import csv
from collections import defaultdict
import json

emotion = {'neutral': 1, 'surprise': 2, 'fear': 3, 'sadness': 4, 'joy': 5, 'disgust': 6, 'anger': 7}

fn2DialogRE = {"train_sent_emo.csv": "train.json", "dev_sent_emo.csv": "dev.json", "test_sent_emo.csv": "test.json"}

def data_processing(file_name):
    data_Utterance = defaultdict(list)
    data_Speaker = defaultdict(list)
    data_Emotion = defaultdict(list)
    data_Utterance_ID = defaultdict(list)

    with open(file_name, 'r') as csvfile:
        rdr = csv.DictReader(csvfile)
        for i in rdr:
            data_Utterance[i['Dialogue_ID']].append(i['Utterance'])
            data_Speaker[i['Dialogue_ID']].append(i['Speaker'])
            data_Emotion[i['Dialogue_ID']].append(i['Emotion'])
            data_Utterance_ID[i['Dialogue_ID']].append(i['Utterance_ID'])

    total_data = list()

    for k in data_Utterance.keys():
        speaker2speakerid = dict()
        for spe in data_Speaker[k]:
            if spe not in speaker2speakerid:
                speaker2speakerid[spe] = str(len(speaker2speakerid) + 1)
        dialog = list()
        relation = list()
        for idx in range(len(data_Utterance[k])):
            rela = dict()
            turn = "Speaker " + speaker2speakerid[data_Speaker[k][idx]] + ": "
            turn = turn + data_Utterance[k][idx]
            dialog.append(turn)
            rela["x"] = "Speaker " + speaker2speakerid[data_Speaker[k][idx]]
            rela["y"] = data_Utterance[k][idx]
            rela["r"] = [data_Emotion[k][idx]]
            rela["rid"] = [emotion[data_Emotion[k][idx]]]
            relation.append(rela)
        
        total_data.append([dialog, relation])


    with open(fn2DialogRE[file_name], 'w', encoding='utf-8') as make_file:
        json.dump(total_data, make_file, indent=1)

data_processing("train_sent_emo.csv")
data_processing("dev_sent_emo.csv")
data_processing("test_sent_emo.csv")



