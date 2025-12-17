from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""

    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])

        prediction = model.predict(data)[0]

        # SAFE confidence calculation
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data).max() * 100
            confidence = f"{prob:.2f}%"
        else:
            confidence = "N/A"

        result = "Spam ❌" if prediction == 1 else "Not Spam ✅"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
