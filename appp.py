from flask import Flask, render_template, url_for, request, session, redirect, flash
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import pyrebase
import bcrypt
from collections.abc import MutableMapping
from google_trans_new import google_translator
tr = google_translator()

# from googletrans import Translator
# tr = Translator()


print(os.listdir())
# so in heroku , it is running this file from the root but not from the server folder
# Hence all the paths will have to be from the root instead from the actual server paths
# So all the paths in this code are like that.

# Flask Config
# But, for Flask Paths below, it runs from inside the server folder hence it follows normal path conventions.
application = app = Flask(__name__, static_folder='/Users/liviagonsalves/Desktop/NLP_proj/client/static',
                          template_folder="/Users/liviagonsalves/Desktop/NLP_proj/client/templates")

app.config["SECRET_KEY"] = "ursecretkey"

ENV = 'dev'

if ENV == 'dev':
    app.debug = True
else:
    app.debug = False

# Our web app's Firebase configuration
firebaseConfig = {
  "apiKey": "AIzaSyC8MYDMZQRqbsCWEXLKKYE7byICf620XXk",
  "authDomain": "nlpproj-b7b4c.firebaseapp.com",
  "databaseURL": "https://nlpproj-b7b4c-default-rtdb.firebaseio.com/",
  "projectId": "nlpproj-b7b4c",
  "storageBucket": "nlpproj-b7b4c.appspot.com",
  "messagingSenderId": "358771080072",
  "appId": "1:358771080072:web:f2dc97dabe7ab5e5c3583d",
  "measurementId": "G-Z80CV42DNR"
};

# Firebase config
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()


# Custom stopwords

with open('/Users/liviagonsalves/Desktop/NLP_proj/server/helper/english-new', 'r') as f:
    stop = f.readlines()
    f.close()
stop = [i.rstrip() for i in stop]


def update_in_db(user, sentence, sentiment):
    try:
        user_data = db.child("users").child(str(user)).get()
        for i in user_data.each():
            if(i.key() == 'sentiments'):
                i.val().append([sentence, sentiment])
                val = i.val()
        db.child("users").child(user).update({"sentiments": val})
        return True
    except Exception as e:
        return("Something Went Wrong !!")


def create_user(user, data):
    try:
        db.child("users").child(str(user)).set(data)
        return (f"{user} Created !")
    except Exception as e:
        return("Something Went Wrong !!")


def fetch_data(user):
    user_Data = db.child("users").child(str(user)).get()
    for i in user_Data.each():
        if(i.key() == 'sentiments'):
            ss = i.val()
    del ss[0]
    ss.reverse()
    return ss


@app.route('/')
def index():
    return render_template('log_reg.html', res=3)


@app.route('/landing')
def landing():
    if('uname' not in session):
        return(render_template('log_reg.html'))
    else:
        ss = fetch_data(session['uname'])
        return render_template('index.html', res=3, ss=ss, lenss=len(ss))


@app.route('/detect_sentiment', methods=['GET', 'POST'])
def detect_sentiment():
    if "uname" in session:
        inp = request.form["inp"]
        print(inp)
        # print(stop)
        # eng = tr.translate(inp).text
        eng = tr.translate(inp)
        eng = eng.lower().replace('\W+', " ").replace("'", " ")
        removed_stopword = []
        for word in eng.split():
            if word not in stop:
                removed_stopword.append(word)
        # eng = np.array([" ".join(removed_stopword)])
        eng = [" ".join(removed_stopword)]
        print(eng)
        tfidfconverter = TfidfVectorizer(
            max_features=200, min_df=1, max_df=0.10)
        tfidf_model = pickle.load(open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/tfidf.pkl', 'rb'))
        x = tfidf_model.transform(eng).toarray()
        models = ['lr.pkl', 'dt.pkl', 'gnb.pkl',
                  'knn.pkl', 'rfc.pkl', 'svm.pkl']
        for i in models:
            model = pickle.load(open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/'+i, 'rb'))
            print(i, " ", model.predict(x))
        model = pickle.load(open('/Users/liviagonsalves/Desktop/NLP_proj/training/models/rfc.pkl', 'rb'))
        pred = model.predict(x)
        print(pred)

        res = 0
        if(pred[0] == 1):
            res = 1
            db_pred = "Happy"
        elif(pred[0] == 0):
            res = 0
            db_pred = "Neutral"
        elif(pred[0] == -1):
            res = -1
            db_pred = "Sad"
        update_in_db(session['uname'], str(inp), str(db_pred))
        ss = fetch_data(session['uname'])
        return render_template('index.html', res=res, ss=ss, lenss=len(ss))
    else:
        return(render_template('log_reg.html'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if(request.method == "POST"):
        uname = request.form['uname']
        passw = request.form['pass']
        if(db.child('users').child(uname).get().val() is not None):

            user_pass = db.child("users").child(
                str(uname)).child("password").get().val()
            if bcrypt.hashpw(passw.encode('utf-8'), bytes(user_pass.replace("b'", "").replace("'", ""), 'utf-8')) == bytes(user_pass.replace("b'", "").replace("'", ""), 'utf-8'):
                session['uname'] = request.form['uname']
                return redirect(url_for('landing'))
            else:
                print("Please enter correct password")
                return(render_template('log_reg.html', err_flag_pass=True))
        else:
            print("User Not found")
            return(render_template('log_reg.html', err_flag_unf=True))
    else:
        return(render_template('log_reg.html'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if(request.method == "POST"):
        uname = request.form['uname']
        passw = request.form['pass']
        email = request.form['email']
        if(db.child('users').child(uname).get().val() is not None):
            print("User existing Please try something else!")
            return(render_template('log_reg.html', err_flag=True))
        else:
            hashpass = bcrypt.hashpw(
                request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            data = {"email": email, "username": uname, "password": str(
                hashpass), "sentiments": [('Sentences', 'Sentiment_val')]}
            create_user(uname, data)
            return redirect(url_for('login'))
    else:
        return(render_template('log_reg.html'))


@app.route('/logout')
def logout():
    session.pop('uname', None)
    return redirect('/')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
