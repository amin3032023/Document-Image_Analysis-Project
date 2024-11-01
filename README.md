# Keyword Spotting (DIA-project)

## Description
This project aims to retrieve words that are identical to the word given in input by the user.
F1-score and Precision/Recall curve is calculated, so the user can see the quality of the result.

## Installation
To install the required Python dependencies, run the following commands:

```bash
sudo apt install python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-cache-dir
```
The flag '--no-cache-dir' prevents the process to be killed if running out of RAM.

Note that this setup has only been tested on Linux Mint 21.3, which is a Debian based distribution.
## Usage
Run the **setup.py** script and wait for the documents to be pre-processed.
The process takes approximately 7 minutes in total.
Information are display in the terminal as shown below:

```text
Documents are not binarized yet. Binarizing them now... (time duration: approx. 2 sec)
Binarization process finished in 2 seconds
Words are not cropped yet. Cropping them now... (time duration: approx. 7 min)
All words of document #1 have been cropped
All words of document #2 have been cropped
[...]
All words of document #15 have been cropped
Cropping process finished in 425 seconds
```

When the setup is ready, run the **main.py** script. The terminal will ask for a word to retrieve:

```text
Please enter the word to search for:
```
Give any word to retrieve. If the word does not appear once among all files, it will not work. Rerun the script with another word.

Giving a valid word will start the process. It takes approximately 5 minutes to finish the process, depending on the word to retrieve. A frequently used word will take more time to process all the data, as the target set will be bigger.

When the process is finished, an F1 score is displayed on the terminal. Additionally, a Precision-Recall curve is saved in documents/results folder, so the user can visualize the quality of the result.

