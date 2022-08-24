from flask import Flask, render_template, request

import torch

import numpy as np
import joblib

app = Flask(__name__)

def preprocess_data(data):
    X = []
    lot_shape_dict = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
    # Dictionary --> np.array
    for k, v in data.items():
        if k =='LotShape':
            if v == 'Reg':
                X.append(4)
            elif v == 'IR1':
                X.append(3)
            elif v == 'IR2':
                X.append(2)
            elif v == 'IR3':
                X.append(1)
        else:
            X.append(float(v))
    
    X = np.array(X)
    X = X.reshape((1, -1))
    print(X.shape)
    
    x_min_max_scaled = joblib.load('./tmp/x_min_max_scaler.save')
    scaled_X = x_min_max_scaled.transform(X)
    scaled_X = torch.FloatTensor(scaled_X)
    
    return scaled_X
    
@app.route('/')
def predict():
    # return "<h1>This is your Flask Server.<h1>"
    return render_template('submit_form.html')


@app.route('/result', methods=['POST'])
def result():
    message=''
    message += '<h1>House Price</h1>'
    message += 'This page will be your prediction form <br>'
    
    # Read data 
    data = request.form
    ## console output
    for k, v in data.items():
        print(k, v)
        message += f'{k}: {v}'
        message += '<br>'
        
    # Preprocess data
    X = preprocess_data(data) # shape= (1, 8)
    
    # Model predict
    model = torch.load('./tmp_pytorch/model.pt')
    y_min_max_scaler = joblib.load('./tmp/y_min_max_scaler.save')
    
    with torch.no_grad():
        model.eval()
        pred = model(X)
        
    pred = y_min_max_scaler.inverse_transform(pred) # shape = (1, 1)
    
    # return prediction
    message += '<br>'
    message += f"Prediction Price: {str(pred[0][0])}"    
    
    return message
app.run(port=4444, debug=True)