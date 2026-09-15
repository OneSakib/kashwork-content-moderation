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
    abusive = abusive_model.predict(abusive_vector)[0]
    abusive_score = abusive_model.predict_proba(abusive_vector)[0][1]
    restricted = restricted_model.predict(restricted_vector)[0]
    restricted_score = restricted_model.predict_proba(restricted_vector)[0][1]
    # print("abusive", abusive)
    # print("restricted", restricted)
    return {
        "abusive": abusive,
        "abusive_score": round(abusive_score, 2),
        "restricted": restricted,
        "restricted_score": round(restricted_score, 2),
    }


if __name__ == "__main__":
    print("PREDICY:", predict("Premium laptop"))
