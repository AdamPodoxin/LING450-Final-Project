# LING 450 final project: interview pass or fail prediction model

## Adam Podoxin, Kyana Sohangar, Elijah Lazar

In this project, we create a machine learning model that predicts whether a candidate passes or fails an interview based on the transcript. We use an appraisal analysis approach developed by [Asher et al. (2009)](https://doi.org/10.1075/li.32.2.10ash). Our data set comes from [Yaswanth Kumar Yallapu's AI Recruitment Pipeline Dataset](https://www.kaggle.com/datasets/yaswanthkumary/ai-recruitment-pipeline-dataset/data). The interviews are AI generated, which limits our ability to accurately judge if this is practical for real interviews. However, the main point of the project is to implement linguistic analysis techniques in a practical way rather than getting the best accuracy. The dataset also additionally includes AI-generated resumes, but we will not be using those.

# Requirements

Python 3.12 is recommended.

Packages:

- `numpy`
- `pandas`
- `spacy`
- `textblob`
- `sklearn`

# Pipeline

## Clean data

The transcripts found in `data/dataset.csv` vary wildly in format. We need to clean them up so it's easier to extract features from them.
Specifically, we want to remove any header/footer metadata from the transcripts, and standardize the interviewer and candidate lines.
To do this, run the command

```bash
python3 pipeline/clean_data.py <input_file> <output_file>
```

This will format the transcripts like so:

```
INTERVIEWER: says something...
CANDIDATE: says something...
INTERVIEWER: says something...
CANDIDATE: says something...
```

Note that although there are many different rules we built in for properly extracting the intreviewer and candidate speech, there are some cases where we were not able to do this for a small number of transcripts.
The number of removed transcripts is printed to the command line when the script finishes.

## Get docs

We use spaCy to do some of our analysis, so we need to convert our transcripts to spaCy Docs.
This step is quite resource-intensive and takes some time, so we can do this once, serialize them to a file, and then load them whenever we need to extract features later.
To do this, run the command

```bash
python3 pipeline/get_docs.py <input_file> <output_folder>
```

All of the required data is saved to the output folder (e.g. `data/transcript_docs`).

## Extract features

We extract features relevant to appraisal analysis.
To do this on the (cleaned) data, run the command

```bash
python3 pipeline/extract_features.py <input_file> <doc_bins_folder> <output_file>
```

The input file is a csv containing the interview transcripts (e.g. `data/dataset_cleaned.csv`) and the output file will contain all the features we use for training the model.
The doc bins folder is the folder where you saved the transcript Docs from the previous step.

## Train the model

Now with the features extracted, we can train the model using the command

```bash
python3 pipeline/train_model.py <input_file>
```

The input file is the one with the features from the extraction step (e.g. `data/data_with_features.csv`). This step will print the classification report of the model after training.
