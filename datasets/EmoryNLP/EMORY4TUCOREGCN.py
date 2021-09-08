import json

emotions = {"Sad": 1, "Mad": 2, "Scared": 3, "Powerful": 4, "Peaceful": 5, "Joyful": 6, "Neutral": 7}

fn2DialogRE = {"emotion-detection-trn.json": "train.json", "emotion-detection-dev.json": "dev.json", "emotion-detection-tst.json": "test.json"}

def data_processing(file_name):
    with open(file_name, "r", encoding="utf8") as f:
        data = json.load(f)

    result_data = []

    for epi in data["episodes"]:
        for sc in epi["scenes"]:
            speakerid = {}
            dialog = []
            relation_all = []
            for ut in sc["utterances"]:
                relation = dict()
                this_spk = []
                for sp in ut["speakers"]:
                    if sp not in speakerid:
                        speakerid[sp] = "Speaker " + str(len(speakerid)+1)
                    this_spk.append(speakerid[sp])
                turn = ", ".join(this_spk) + ": " + ut["transcript"]
                dialog.append(turn)
                relation["x"] = ", ".join(this_spk)
                relation["y"] = ut["transcript"]
                relation["r"] = [ut["emotion"]]
                relation["rid"] = [emotions[ut["emotion"]]]
                relation_all.append(relation)
            result_data.append([dialog, relation_all])

    with open(fn2DialogRE[file_name], 'w', encoding='utf-8') as make_file:
        json.dump(result_data, make_file, indent=1)


data_processing("emotion-detection-trn.json")
data_processing("emotion-detection-dev.json")
data_processing("emotion-detection-tst.json")

