import joblib
from src.preprocessing import normalize_text

abusive_model = joblib.load("models/abusive_model.pkl")

abusive_vectorizer = joblib.load("models/abusive_vectorizer.pkl")

restricted_model = joblib.load("models/restricted_model.pkl")

restricted_vectorizer = joblib.load("models/restricted_vectorizer.pkl")


def predict(text):

    clean_text = normalize_text(text)

    abusive_vector = abusive_vectorizer.transform([clean_text])

    restricted_vector = restricted_vectorizer.transform([clean_text])

    abusive_score = abusive_model.predict_proba(abusive_vector)[0][1]

    restricted_score = restricted_model.predict_proba(restricted_vector)[0][1]

    return {
        "abusive_score": round(abusive_score, 2),
        "restricted_score": round(restricted_score, 2),
    }


if __name__ == "__main__":
    print("PREDICY:", predict("Premium laptop"))
