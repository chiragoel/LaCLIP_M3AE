
# Data Augmentation Utility

This Python script, `augment_data.py`, is designed to process and augment text data using the OpenAI API. The script rewrites sentences from a JSON dataset, altering their sentiment to create an augmented dataset that might be useful for training machine learning models.

## Features

- **Read and Write JSON**: Read data from a JSON file, and write the augmented data back to a JSON file.
- **Data Sampling**: Sample a subset of the data from the original dataset if needed.
- **Sentiment Rewriting**: Utilizes the OpenAI API to rewrite sentences by changing their sentiment, either to sarcasm or its opposite.
- **API Integration**: Requires an OpenAI API key to function, which should be placed in a text file.

## Setup

1. **Install Dependencies**:
    ```
    pip install openai
    ```

2. **API Key**:
    Place your OpenAI API key in a text file. Update the script to read the API key from this file.

## Usage

- Place your input JSON file containing the original sentences in the project directory.
- Run the script:
  ```
  python augment_data.py --path_api_key=</path/to/api/key> --path_training_dataset=</path/to/training/dataset/> --save_path=</path/to/save/augmented/data>
  ```

## Input and Output

- **Input**: A JSON file with sentences.
- **Output**: A JSON file (`aug_test.json`) with the augmented data.

## Example

- Input Sentence: "He works very hard to barely meet deadlines."
- Output Sentence: "Oh, he is a real superhero, saving the day by just barely meeting his deadlines."

## Note

This script uses OpenAI's API and will incur API usage costs according to OpenAI's pricing model.
