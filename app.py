from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def predict():
	if request.method == 'GET':
		return render_template('index.html')

	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	
	df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
	df['message'] = df['v2']
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	bow_transformer = CountVectorizer()
	X = bow_transformer.fit_transform(df['message'])

	#Load the model that was created initially
	spam_model = open('our_model.pkl','rb')
	clf = joblib.load(spam_model)

	if request.method == 'POST':
		message = request.form['sample_text']
		data = [message]
		vect = bow_transformer.transform(data).toarray()
		my_prediction = clf.predict(vect)

		my_prediction = str(my_prediction[0])
		if my_prediction == '1':
			return "That looks like a spam message"
		else:
			return "This looks like a ham message."



if __name__ == '__predict__':
	app.run(debug=True)