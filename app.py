from flask import Flask, request, render_template
import joblib, re, nltk
from nltk.corpus import stopwords

nltk.download('stopwords') #comment out when installed

app = Flask(__name__)
model=joblib.load('model.pkl') #load moeel
vector=joblib.load('vectorizer.pkl')#load vectorizer

categories = {
    0: ('Not Offensive', None),
    1: ('Offensive ','General Offensive'),
    3: ('Offensive ','Hate Speech'),
    4: ('Offensive ','Insult')
}
def clean(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))|set(stopwords.words('indonesian')) #gabingin stopword indo sama inggris
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = ''
    confidence = None
    type = None
    if request.method == 'POST':
        user_input = request.form['text']
        vec = vector.transform([clean(user_input)])
        prediction = model.predict(vec)[0]
        result, type = categories[prediction]
        confidence = round(model.predict_proba(vec).max() * 100, 2)
    return render_template('index.html', result=result, user_input=user_input, confidence=confidence, type=type)
import os
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
