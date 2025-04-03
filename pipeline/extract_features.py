import sys
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin
from textblob import TextBlob
from senticnet.senticnet import SenticNet


sentic = SenticNet()


num_processes_to_use = cpu_count() // 2


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


def get_sentiment_stats_row(row: pd.Series):
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


def get_sentiment_stats_chunk(rows: list[list[pd.Series]]):
    return [get_sentiment_stats_row(row) for row in rows]


def get_sentiment_stats(data: pd.DataFrame):
    num_chunks = num_processes_to_use
    chunk_size = int(np.ceil(len(data) / num_chunks))
    sentiment_stats_list = []

    with Pool(num_chunks) as pool:
        chunks = [
            [row for _, row in data.iloc[i:i + chunk_size].iterrows()]
            for i in range(0, len(data), chunk_size)
        ]
        sentiment_stats_list = pool.map(get_sentiment_stats_chunk, chunks)

    sentiment_stats = pd.DataFrame([item for sublist in sentiment_stats_list for item in sublist])
    return sentiment_stats


def get_sentics_for_word(word: str):
    try:
        concept = sentic.concept(word)

        sentics: dict[str, float] = concept["sentics"]
        emotions = {f"sentic_emotion_{key}": float(value) for key, value in sentics.items()}
        
        moodtags: list[str] = concept["moodtags"]
        moods = {f"sentic_mood_{moodtag[1:]}": 1 for moodtag in moodtags}

        return emotions, moods
    except:
        return None, None


def get_sentic_ratios(doc: Doc):
    sentic_sums: dict[str, float] = {}

    for token in doc:
        emotions, moods = get_sentics_for_word(token.lemma_)

        if not emotions or not moods:
            continue

        for key, value in emotions.items():
            sentic_sums[key] = sentic_sums.get(key, 0) + value
        
        for key, value in moods.items():
            sentic_sums[key] = sentic_sums.get(key, 0) + value
        
    doc_len = len(doc)
    sentic_ratios = {f"{key}_ratio": value / doc_len for key, value in sentic_sums.items()}

    return pd.Series(sentic_ratios)


def get_docs_from_disk(doc_bins_folder: str):
    nlp = spacy.load("en_core_web_sm").from_disk(f"{doc_bins_folder}/nlp")

    full_transcript_doc_bin = DocBin().from_disk(f"{doc_bins_folder}/full.spacy")
    interviewer_transcript_doc_bin = DocBin().from_disk(f"{doc_bins_folder}/interviewer.spacy")
    candidate_transcript_doc_bin = DocBin().from_disk(f"{doc_bins_folder}/candidate.spacy")

    return pd.DataFrame({
        "full_transcript_doc": [doc for doc in full_transcript_doc_bin.get_docs(nlp.vocab)],
        "interviewer_transcript_doc": [doc for doc in interviewer_transcript_doc_bin.get_docs(nlp.vocab)],
        "candidate_transcript_doc": [doc for doc in candidate_transcript_doc_bin.get_docs(nlp.vocab)],
    })


ignore_columns = ["Name", "Role", "Transcript", "Resume", "Reason_for_decision", "Job_Description"]


def extract_all_features(data: pd.DataFrame):
    print("Getting appraisal category ratios...")

    full_appraisal_ratios = data["full_transcript_doc"].apply(get_appraisal_ratios)
    full_appraisal_ratios = get_ratios_with_renamed_columns(full_appraisal_ratios, "full")

    interviewer_appraisal_ratios = data["interviewer_transcript_doc"].apply(get_appraisal_ratios)
    interviewer_appraisal_ratios = get_ratios_with_renamed_columns(interviewer_appraisal_ratios, "interviewer")

    candidate_appraisal_ratios = data["candidate_transcript_doc"].apply(get_appraisal_ratios)
    candidate_appraisal_ratios = get_ratios_with_renamed_columns(candidate_appraisal_ratios, "candidate")


    print("Getting attitudinal adjective ratios...")
    
    full_attitudinal_ratios = data["full_transcript_doc"].apply(get_attitudinal_ratios)
    full_attitudinal_ratios = get_ratios_with_renamed_columns(full_attitudinal_ratios, "full")
    
    interviewer_attitudinal_ratios = data["interviewer_transcript_doc"].apply(get_attitudinal_ratios)
    interviewer_attitudinal_ratios = get_ratios_with_renamed_columns(interviewer_attitudinal_ratios, "interviewer")

    candidate_attitudinal_ratios = data["candidate_transcript_doc"].apply(get_attitudinal_ratios)
    candidate_attitudinal_ratios = get_ratios_with_renamed_columns(candidate_attitudinal_ratios, "candidate")


    print("Getting sentiment statistics...")

    sentiment_stats = get_sentiment_stats(data)


    print("Getting sentic emotion and mood ratios...")

    full_sentic_ratios = data["full_transcript_doc"].apply(get_sentic_ratios).fillna(0)

    print("Cleaning up...")

    feature_columns = [
        data,
        full_appraisal_ratios,
        interviewer_appraisal_ratios,
        candidate_appraisal_ratios,
        full_attitudinal_ratios,
        interviewer_attitudinal_ratios,
        candidate_attitudinal_ratios,
        sentiment_stats,
        full_sentic_ratios
    ]

    data_with_features = pd.concat(feature_columns, axis=1).fillna(0)

    data_with_features.drop(columns=ignore_columns, axis=1, inplace=True)

    return data_with_features


def main(input_file: str, doc_bins_folder: str, output_file: str):
    print("Reading from", input_file)
    data = pd.read_csv(input_file)

    print("Getting docs from", doc_bins_folder)
    transcript_docs = get_docs_from_disk(doc_bins_folder)
    data_with_transcript_docs = pd.concat([data, transcript_docs], axis=1)

    print("Extracting features...")
    start_time = time.time()
    data_with_features = extract_all_features(data_with_transcript_docs)

    end_time = time.time()
    print("Done in", end_time - start_time, "seconds. Saving to", output_file)
    data_with_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 pipeline/extract_features.py <input_file> <doc_bins_folder> <output_file>")
    
    else:
        input_file = sys.argv[1]
        doc_bins_folder = sys.argv[2]
        output_file = sys.argv[3]

        main(input_file, doc_bins_folder, output_file)