import sys
import pandas as pd


def extract_all_features(data):
    return data


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = pd.read_csv(input_file)

    data_with_features = extract_all_features(data)

    data_with_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/extract_features.py <input_file> <output_file>")
    else:
        main()