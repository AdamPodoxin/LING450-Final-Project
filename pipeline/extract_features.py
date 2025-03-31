import sys
import time
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc
from textblob import TextBlob


nlp = spacy.load("en_core_web_sm")


def separate_interviewer_and_candidate_transcripts(row: pd.Series):
    transcript: str = row["Transcript"]

    lines = transcript.split("\n")
    interviewer_lines = [line for line in lines if line.startswith("INTERVIEWER:")]
    candidate_lines = [line for line in lines if line.startswith("CANDIDATE:")]

    full_transcript = "\n".join([line.split(": ")[1] for line in lines])
    interviewer_transcript = "\n".join([line.split(": ")[1] for line in interviewer_lines])
    candidate_transcript = "\n".join([line.split(": ")[1] for line in candidate_lines])

    result_dict = {
        "full_transcript": full_transcript, 
        "interviewer_transcript": interviewer_transcript, 
        "candidate_transcript": candidate_transcript
    }

    return pd.Series(data=result_dict)


def get_transcript_docs(transcripts: pd.Series):
    with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
        docs = list(nlp.pipe(transcripts, n_process=4))
        
    return docs


def get_data_with_transcript_docs(data: pd.DataFrame):
    separated_transcripts = data.apply(separate_interviewer_and_candidate_transcripts, axis=1)

    full_transcript_docs = get_transcript_docs(separated_transcripts["full_transcript"])
    interviewer_transcript_docs = get_transcript_docs(separated_transcripts["interviewer_transcript"])
    candidate_transcript_docs = get_transcript_docs(separated_transcripts["candidate_transcript"])

    transcript_docs_dict = {
        "full_transcript_doc": full_transcript_docs,
        "interviewer_transcript_doc": interviewer_transcript_docs,
        "candidate_transcript_doc": candidate_transcript_docs,
    }

    transcript_docs_df = pd.DataFrame(data=transcript_docs_dict)

    return pd.concat([data, transcript_docs_df], axis=1)


def get_word_category_ratios(doc: Doc, categories: dict[str, list[str]]):
    word_to_category = {word: category 
                for category, words in categories.items() 
                for word in words}

    word_counts = {word: 0 for word in word_to_category.keys()}
    
    for token in doc:
        if token.lemma_ in word_to_category:
            word_counts[token.lemma_] += 1
    
    category_counts = {category: sum([word_counts[word] 
                                      for word, cat in word_to_category.items() 
                                      if cat == category]) 
                                      for category in categories.keys()}
    
    ratios = {f"{category}_ratio": category_counts[category] / len(doc) 
              for category in categories.keys()}
    
    return ratios


# From Biber (2004)
attitudinal_categories = {
    "positive": ["amazed", "amazing", "amused", "funny", "glad", "good", 
                 "grateful", "great", "happy", "hopeful", "wonderful", 
                 "pleased", "preferable", "reassured", "relieved", "thankful", 
                 "extraordinary", "incredible", "acceptable", "advisable", 
                 "appropriate", "encouraged", "silly", "desirable", "fortunate", 
                 "interesting", "nice", "lucky", "satisfied", "sensible", 
                 "surprised", "surprising", "neat"],
    "neutral": ["adamant", "anomalous", "aware", "careful", "conceivable", 
                "critical", "crucial", "curious", "essential", "fitting", 
                "imperative", "incidental", "inconceivable", "indisputable", 
                "ironic", "natural", "necessary", "notable", "noteworthy", 
                "noticeable", "obligatory", "odd", "okay", "paradoxical", 
                "peculiar", "ridiculous", "strange", "sufficient", "typical", 
                "unaware", "understandable", "untypical", "unusual", "vital"],
    "negative": ["afraid", "alarmed", "angry", "annoyed", "annoying", "concerned", 
                 "depressed", "upset", "upsetting", "worried", "disappointed", 
                 "disappointing", "dissatisfied", "distressed", "disturbed", 
                 "dreadful", "embarrasing", "uncomfortable", "unfair", "unfortunate", 
                 "unhappy", "unlucky", "awful", "frightened", "frightening", "horrible", 
                 "hurt", "unacceptable", "unthinkable", "ashtonished", "ashtonishing", "sorry", 
                 "mad", "stupid", "irritated", "irritating", "sad", "shocked", "shocking", "tragic"],
}


def get_attitudinal_ratios(doc: Doc):
    ratios = get_word_category_ratios(doc, attitudinal_categories)
    ratios = {f"attitudinal_{column}": value for column, value in ratios.items()}
    return pd.Series(ratios)


# From Asher et al. (2009)
appraisal_categories = {
    "inform": ["inform", "notify", "explain"],
    "assert": ["assert", "claim", "insist"],
    "tell": ["say", "announce", "report"],
    "remark": ["comment", "observe", "remark"],
    "think": ["think", "reckon", "consider"],
    "guess": ["presume", "suspect", "wonder"],
    "blame": ["blame", "criticize", "condemn"],
    "praise": ["praise", "agree", "approve"],
    "appreciation": ["good", "shameful", "brilliant"],
    "recommend": ["advise", "argue for"],
    "suggest": ["suggest", "propose"],
    "hope": ["wish", "hope"],
    "anger_calmdown": ["irritation", "anger"],
    "astonishment": ["astound", "daze"],
    "love_fascinate": ["fascinate", "captivate"],
    "hate_disappoint": ["demoralize", "disgust"],
    "fear": ["fear", "frighten", "alarm"],
    "offense": ["hurt", "chock"],
    "sadness_joy": ["happy", "sad"],
    "bore_entertain": ["bore", "distraction"]
}


def get_appraisal_ratios(doc: Doc):
    ratios = get_word_category_ratios(doc, appraisal_categories)
    ratios = {f"appraisal_{column}": value for column, value in ratios.items()}
    return pd.Series(ratios)


def get_ratios_with_renamed_columns(ratios_df: pd.DataFrame, prefix: str):
    renamed_columns = {column: f"{prefix}_{column}" for column in ratios_df.columns}
    return ratios_df.rename(columns=renamed_columns)


def get_sentiment_stats(row: pd.Series):
    full_transcript_doc: Doc = row["full_transcript_doc"]
    interviewer_transcript_doc: Doc = row["interviewer_transcript_doc"]
    candidate_transcript_doc: Doc = row["candidate_transcript_doc"]

    full_transcript_blob = TextBlob(full_transcript_doc.text)
    interviewer_transcript_blob = TextBlob(interviewer_transcript_doc.text)
    candidate_transcript_blob = TextBlob(candidate_transcript_doc.text)

    return pd.Series(data={
        "overall_full_sentiment": full_transcript_blob.sentiment.polarity,
        "overall_interviewer_sentiment": interviewer_transcript_blob.sentiment.polarity,
        "overall_candidate_sentiment": candidate_transcript_blob.sentiment.polarity,
        
        "full_sentiment_trend": 
            full_transcript_blob.sentences[-1].sentiment.polarity - full_transcript_blob.sentences[0].sentiment.polarity,
        "interviewer_sentiment_trend": 
            interviewer_transcript_blob.sentences[-1].sentiment.polarity - interviewer_transcript_blob.sentences[0].sentiment.polarity,
        "candidate_sentiment_trend": 
            candidate_transcript_blob.sentences[-1].sentiment.polarity - candidate_transcript_blob.sentences[0].sentiment.polarity,
        
        "full_sentiment_variability": 
            np.std([sentence.sentiment.polarity for sentence in full_transcript_blob.sentences]),
        "interviewer_sentiment_variability": 
            np.std([sentence.sentiment.polarity for sentence in interviewer_transcript_blob.sentences]),
        "candidate_sentiment_variability": 
            np.std([sentence.sentiment.polarity for sentence in candidate_transcript_blob.sentences]),
    })


ignore_columns = ["Name", "Role", "Transcript", "Resume", "Reason_for_decision", "Job_Description"]


def extract_all_features(data: pd.DataFrame):
    print("Processing transcripts...")

    data_with_transcript_docs = get_data_with_transcript_docs(data)


    print("Getting appraisal category ratios...")

    full_appraisal_ratios = data_with_transcript_docs["full_transcript_doc"].apply(get_appraisal_ratios)
    full_appraisal_ratios = get_ratios_with_renamed_columns(full_appraisal_ratios, "full")

    interviewer_appraisal_ratios = data_with_transcript_docs["interviewer_transcript_doc"].apply(get_appraisal_ratios)
    interviewer_appraisal_ratios = get_ratios_with_renamed_columns(interviewer_appraisal_ratios, "interviewer")

    candidate_appraisal_ratios = data_with_transcript_docs["candidate_transcript_doc"].apply(get_appraisal_ratios)
    candidate_appraisal_ratios = get_ratios_with_renamed_columns(candidate_appraisal_ratios, "candidate")


    print("Getting attitudinal adjective ratios...")
    
    full_attitudinal_ratios = data_with_transcript_docs["full_transcript_doc"].apply(get_attitudinal_ratios)
    full_attitudinal_ratios = get_ratios_with_renamed_columns(full_attitudinal_ratios, "full")
    
    interviewer_attitudinal_ratios = data_with_transcript_docs["interviewer_transcript_doc"].apply(get_attitudinal_ratios)
    interviewer_attitudinal_ratios = get_ratios_with_renamed_columns(interviewer_attitudinal_ratios, "interviewer")

    candidate_attitudinal_ratios = data_with_transcript_docs["candidate_transcript_doc"].apply(get_attitudinal_ratios)
    candidate_attitudinal_ratios = get_ratios_with_renamed_columns(candidate_attitudinal_ratios, "candidate")


    print("Getting sentiment statistics...")

    sentiment_stats = data_with_transcript_docs.apply(get_sentiment_stats, axis=1)


    print("Cleaning up...")

    data_with_features = pd.concat([
        data,
        full_appraisal_ratios,
        interviewer_appraisal_ratios,
        candidate_appraisal_ratios,
        full_attitudinal_ratios,
        interviewer_attitudinal_ratios,
        candidate_attitudinal_ratios,
        sentiment_stats
    ], axis=1)

    data_with_features.drop(columns=ignore_columns, axis=1, inplace=True)

    return data_with_features


def main(input_file: str, output_file: str):
    print("Reading from", input_file)
    data = pd.read_csv(input_file)

    print("Extracting features...")
    start_time = time.time()
    data_with_features = extract_all_features(data)

    end_time = time.time()
    print("Done in", end_time - start_time, "seconds. Saving to", output_file)
    data_with_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/extract_features.py <input_file> <output_file>")
    
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        main(input_file, output_file)