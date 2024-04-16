import json
import random

TRAIN_DATA_PATH = "../data/text_json_final/train.json"
VALID_DATA_PATH = "../data/text_json_final/valid.json"
TEST_DATA_PATH = "../data/text_json_final/test.json"
AUG_DATA_PATH = "../data/text_json_final/augmented_data.json"


def combine_train_aug_data(TRAIN_DATA_PATH, AUG_DATA_PATH, how='total', prob=0.5, seed=42):

    random.seed(42)

    train_data = json.load(open(TRAIN_DATA_PATH, 'r'))
    aug_data = json.load(open(AUG_DATA_PATH, 'r'))

    if how=='total':
        #total combine
        total_train_aug_data = train_data + aug_data
        print("total length for all: ", len(total_train_aug_data))
        with open("../data/text_json_final/train_aug_total.json", "w+") as f:
            json.dump(total_train_aug_data, f)

    if how=='random':
        #with x% porb of choosing from aug sample
        train_aug_samples = []

        for i in range(len(train_data)):
            if random.random() < prob: # prob-choos aug sample
                index = random.randint(0, len(aug_data)-1)
                train_aug_samples.append(aug_data[index])
            else: # 1-prob, train sample
                index = random.randint(0, len(train_data)-1)
                train_aug_samples.append(train_data[index])
        print("total length for random: ", len(train_aug_samples))
        with open("../data/text_json_final/train_aug_random.json", "w+") as f:
            json.dump(train_aug_samples, f)


if __name__=="__main__":
    combine_train_aug_data(TRAIN_DATA_PATH, AUG_DATA_PATH, how='total')
    combine_train_aug_data(TRAIN_DATA_PATH, AUG_DATA_PATH, how='random', prob=0.5)


