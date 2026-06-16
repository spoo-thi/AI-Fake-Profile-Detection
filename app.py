from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["profile_pic"]),
        float(request.form["username_ratio"]),
        float(request.form["fullname_words"]),
        float(request.form["fullname_ratio"]),
        float(request.form["name_equal"]),
        float(request.form["description"]),
        float(request.form["external_url"]),
        float(request.form["private"]),
        float(request.form["posts"]),
        float(request.form["followers"]),
        float(request.form["following"])
    ]

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "⚠️ Fake Profile Detected"
    else:
        result = "✅ Real Profile"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)