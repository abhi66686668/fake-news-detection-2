from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None

    if request.method == "POST":

        news = request.form["news"]

        transformed = vectorizer.transform([news])

        result = model.predict(transformed)[0]

        prediction = "REAL NEWS" if result == 1 else "FAKE NEWS"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)