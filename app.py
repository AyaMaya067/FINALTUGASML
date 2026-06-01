from flask import Flask, request, render_template
import joblib, re, nltk
from nltk.corpus import stopwords

nltk.download('stopwords') #comment out when installed

app = Flask(__name__)
model=joblib.load('model.pkl') #load moeel
vector=joblib.load('vectorizer.pkl')#load vectorizer

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
    if request.method == 'POST':
        user_input = request.form['text']
        vec = vector.transform([clean(user_input)])
        prediction = model.predict(vec)[0]
        result = 'Offensive' if prediction == 1 else 'Not Offensive'
    return render_template('index.html', result=result, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=False)