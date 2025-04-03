import sys
import time
from multiprocessing import cpu_count
import pandas as pd
import spacy
from spacy.tokens import DocBin


nlp = spacy.load("en_core_web_sm")


num_processes_to_use = cpu_count() // 2


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
        docs = list(nlp.pipe(transcripts, n_process=num_processes_to_use))
        
    return docs


def get_doc_bins(data: pd.DataFrame):
    separated_transcripts = data.apply(separate_interviewer_and_candidate_transcripts, axis=1)

    full_transcript_docs = get_transcript_docs(separated_transcripts["full_transcript"])
    interviewer_transcript_docs = get_transcript_docs(separated_transcripts["interviewer_transcript"])
    candidate_transcript_docs = get_transcript_docs(separated_transcripts["candidate_transcript"])

    return DocBin(docs=full_transcript_docs), \
            DocBin(docs=interviewer_transcript_docs), \
            DocBin(docs=candidate_transcript_docs)


def main(input_file: str, output_folder: str):
    print("Reading from", input_file)
    data = pd.read_csv(input_file)

    print("Processing transcripts...")
    start_time = time.time()

    full_transcript_doc_bin, \
    interviewer_transcript_doc_bin, \
    candidate_transcript_doc_bin = get_doc_bins(data)

    end_time = time.time()
    print("Done in", end_time - start_time, "seconds. Saving to", output_folder)

    full_transcript_doc_bin.to_disk(f"{output_folder}/full.spacy")
    interviewer_transcript_doc_bin.to_disk(f"{output_folder}/interviewer.spacy")
    candidate_transcript_doc_bin.to_disk(f"{output_folder}/candidate.spacy")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/get_docs.py <input_file> <output_folder>")
    
    else:
        input_file = sys.argv[1]
        output_folder = sys.argv[2]

        main(input_file, output_folder)