from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        msg = request.form["message"]
        data = cv.transform([msg])
        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Spam Message ❌"
        else:
            result = "Not Spam ✅"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
