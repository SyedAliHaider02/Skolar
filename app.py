from flask import Flask, request, render_template, jsonify
import string
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  

class Form(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    keywords = StringField('Keywords', validators=[DataRequired()])
    submit = SubmitField('Predict')

def lower(text):
    return text.lower()

def remove_tags(text):
    tags = ['\n\n', '\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def get_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ['PROPN', 'ADJ', 'NOUN', 'VERB']]
    return keywords

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def compute_tfidf_score(text, keywords):
    corpus = [text] + keywords
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    paper_tfidf = tfidf_matrix[0].toarray().flatten()
    score = sum(paper_tfidf[vectorizer.vocabulary_.get(word, -1)] for word in keywords if word in vectorizer.vocabulary_)
    return score

@app.route('/')
def upload_form():
    form = Form()
    return render_template('home.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    keywords = request.form['keywords'].split()
    text1=text
    # Preprocess text
    text = lower(text)
    text = remove_tags(text)
    text = remove_punct(text)
    text= get_keywords(text)
    text= remove_stopwords(text)
    text=' '.join(text)
    
    score = compute_tfidf_score(text, keywords)
    score=score*100
    
    return render_template('predict.html',score=score,text1=text1,keywords=keywords)

if __name__ == '__main__':
    app.run(debug=True)
