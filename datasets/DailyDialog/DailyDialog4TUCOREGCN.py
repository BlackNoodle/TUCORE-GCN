from collections import defaultdict
import json

emotion = { 0: 'no emotion', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}

def data_processing(data_type):
    total_data = []

    dialog = []

    labels = []

    with open(data_type + '/' + 'dialogues_' + data_type + '.txt', 'r') as data:
        data = data.readlines()
        for d in data:
            dialog.append(d.split('__eou__')[:-1])

    with open(data_type + '/' + 'dialogues_emotion_' + data_type + '.txt', 'r') as data:
        data = data.readlines()
        for d in data:
            labels.append(d.split(" ")[:-1])

    for i in range(len(dialog)):
        di = list()
        relation = list()
        for idx in range(len(dialog[i])):
            rela = dict()
            turn_idx = "1" if idx % 2 == 0 else "2"
            turn = "Speaker " + turn_idx + ": "
            turn = turn + dialog[i][idx]
            di.append(turn)
            rela["x"] = "Speaker " + turn_idx
            rela["y"] = dialog[i][idx]
            rela["r"] = [emotion[int(labels[i][idx])]]
            rela["rid"] = [int(labels[i][idx]) + 1]
            relation.append(rela)
        
        total_data.append([di, relation])
    
    if data_type == "validation":
        with open('dev.json', 'w', encoding='utf-8') as make_file:
            json.dump(total_data, make_file, indent=1)
    else:
        with open(data_type + '.json', 'w', encoding='utf-8') as make_file:
            json.dump(total_data, make_file, indent=1)

data_processing("train")
data_processing("validation")
data_processing("test")