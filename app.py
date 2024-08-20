import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)  #creating app

# import model
scalar=pickle.load(open('scaling.pkl','rb'))
model=pickle.load(open('regmod.pkl','rb'))   #laoding modle

@app.route('/')
# The @app.route decorator specifies the URL pattern for the function it decorates. In this example, accessing the root URL (/) will call the home function
def home():
    return render_template('home.html')
    # he render_template function is used to render an HTML template and pass variables to it.

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)  #this data is from json and is in key value pairs
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted price is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
    