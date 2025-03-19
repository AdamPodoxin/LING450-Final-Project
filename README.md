# LING 450 final project: interview pass or fail prediction model

## Adam Podoxin, Kyana Sohangar, Elijah Lazar

In this project, we create a machine learning model that predicts whether a candidate passes or fails an interview based on the transcript. We use an appraisal analysis approach developed by [Asher et al. (2009)](https://doi.org/10.1075/li.32.2.10ash). Our data set comes from [Yaswanth Kumar Yallapu's AI Recruitment Pipeline Dataset](https://www.kaggle.com/datasets/yaswanthkumary/ai-recruitment-pipeline-dataset/data). The interviews are AI generated, which limits our ability to accurately judge if this is practical for real interviews. However, the main point of the project is to implement linguistic analysis techniques in a practical way rather than getting the best accuracy. The dataset also additionally includes AI-generated resumes, but we will not be using those.

# Requirements

Python 3.12 is recommended.

Packages:

- `pandas`
- `spacy`
- `sklearn`

# Pipeline

## Extract features

We extract features relevant to appraisal analysis. To do this on the data, run the command

```bash
python3 pipeline/extract_features.py <input_file> <output_file>
```

The input file is a csv containing the interview transcripts (e.g. `data/dataset.csv`) and the output file will contain all the features we use for training the model.

## Train the model

Now with the features extracted, we can train the model using the command

```bash
python3 pipeline/train_model.py <input_file>
```

The input file is the one with the features from the extraction step (e.g. `data/data_with_features.csv`). This step will print the classification report of the model after training.
