import sys
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


result_column = "decision"
columns_to_ignore = ["ID", result_column]


def train_model(data: pd.DataFrame):
    X = data.drop(columns=columns_to_ignore, axis=1)
    y = data[result_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importances_df = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])\
                            .sort_values("importance", ascending=False)
    print("\nFeature Importances:")
    print(feature_importances_df)

    feature_importances_df.to_csv("feature_importances.csv", index=True)
    print("\nFeature importances saved to feature_importances.csv")

    return model


def main(input_file: str):
    data_with_features = pd.read_csv(input_file)
    train_model(data_with_features)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pipeline/train_model.py <input_file>")
    else:
        input_file = sys.argv[1]
        main(input_file)