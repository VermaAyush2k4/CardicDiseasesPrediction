# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        # trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form.get('fbs'))
        # restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope'))
        ca = int(request.form['ca'])
        thal = int(request.form.get('thal'))
        modelSelected = int(request.form.get('selectModel'))


        data = np.array([[age,sex,cp,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]])
        filedata = ['heart-disease-prediction-knn-model.pkl' , 'logistic_regression.pkl', 'svm.pkl', 'random_forest.pkl' ,'decision_tree.pkl' ]
        filename = filedata[modelSelected]
        model = pickle.load(open(filename, 'rb'))
        my_prediction = model.predict(data)





    return render_template('result.html', prediction=my_prediction , selection = modelSelected)
        
        

if __name__ == '__main__':
	#app.run(debug=True)
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)
