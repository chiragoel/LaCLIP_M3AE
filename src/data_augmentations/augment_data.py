import json
import random
import argparse

import openai

def read_data_from_json(file_path, sample_size=None):
    with open(file_path, "r") as file:
        data = json.load(file)
        if sample_size:
            data = random.sample(data, min(sample_size, len(data)))
        return data


def read_api_key(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()


def write_data_to_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def rewrite_entity_driven(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in sarcastic sentiment analysis.",
            },
            {
                "role": "user",
                "content": f'''Given a non-sarcastic sentence, 
            identify the main entity or subject in the sentence, 
            analyze the context to understand both implicit and 
            explicit meanings, and convert it into a sarcastic sentence. 
            Use sarcasm techniques such as exaggeration, irony, 
            or rhetorical questions effectively. 
            Adjust the tone to ensure it reflects sarcasm clearly.
            Example:

            Input Sentence: "He works very hard to barely meet deadlines."
            Output: "Oh, he is a real superhero, saving the day by just barely 
            meeting his deadlines."
            Input Sentence: teacher paul basenberg bringing art to life ! what a gift ! mechanical pencil drawing , wow ! where KEY ENTITY: drawing.
            Output Sentence: teacher paul basenberg bringing art to life ! cannot see any color. mechanical pencil drawing ,dumb !
            Input Sentence: the george halas trophy used to look like a real trophy. it looks like a paperweight now . where KEY ENTITY: trophy.
            Output Sentence: wow, it's so impressive to turn the george halas trophy into a paperweight now.
            Input Sentence: pretty girl where KEY ENTITY: girl.
            Output Sentence: not a pretty girl at all  
            Task:
            Convert the following sentence into a sarcastic version by applying the above method:"{text}"''',
            },
        ],
        max_tokens=60,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]


def rewrite_emotion_aware(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Choose the appropriate engine for your needs
        messages=[
            {
                "role": "system",
                "content": "You are an expert in sarcastic sentiment analysis.",
            },
            {
                "role": "user",
                "content": f'''Given a sarcastic sentence, identify the emotional tone 
                and sarcastic elements such as exaggeration or irony and use them 
                to convert this sentence into a strictly straightforward and 
                non-sarcastic sentence. Only give the sentence as output.
                Use opposite words to change the sentiment. 
                Make sure that the sentence re-created is meaningful.
                
                Here are a few examples to help you:
                Input Sentence: "Oh, I just love working late on weekends."
                Output: "I find working late on weekends challenging and prefer not to."
                Input Sentence: i love waiting <num> min for a cab - such shortage ....... <user> please allow uber . this is insane .
                Output: "I hate not being able to get a cab within <num> minutes - there's always a shortage. <user>, can we please use Uber? This is ridiculous."
                
                This is the sentence you have to modify: "{text}"''',
            },
        ],
        temperature=0.0,
        max_tokens=60,
    )
    return response["choices"][0]["message"]["content"]


def main(args):
    api_key = read_api_key(args.path_api_key)
    openai.api_key = api_key
    data = read_data_from_json(args.path_training_dataset)

    augmented_data = []

    for item in data[0:5]:
        sentence = item["text"]
        image_id = item["image_id"]
        label = item["label"]

        if label == 1:
            new_label = 0
            rewritten_text = rewrite_entity_driven(sentence)
            data_entry = {
                "image_id": image_id,
                "text": rewritten_text,
                "label": new_label,
            }
            augmented_data.append(data_entry)
        elif label == 0:
            new_label = 1
            rewritten_text = rewrite_entity_driven(sentence)
            data_entry = {
                "image_id": image_id,
                "text": rewritten_text,
                "label": new_label,
            }
            augmented_data.append(data_entry)

    write_data_to_json(args.save_path, augmented_data)
    print("Augmented data has been written to the respective JSON files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_api_key', type=str, help='Path to the file with your api key')
    parser.add_argument('--path_training_dataset', type=str, help='Path to the training dataset')
    parser.add_argument('--save_path', type=str, help='Path to save augmented data')
    args = parser.parse_args()
    main(args)
