from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

Model=joblib.load('D:\ml\practice\Classification\ServerFile\plks/classifier_cc.pkl')

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/predict')
def predict():
    try:
        data = np.array([[2.53634674, 0.16648011, 0.37871832, 0.73279341, 0.91574848,
       0.0686684 , 0.12853936, 0.02759383, 0.497198  , 0.2514121 ,
       0.36304944, 0.09079417, 0.2514121 , 0.06692807, 0.46238896,
       0.0986979 , 0.36378697, 0.09090954, 0.2514121 , 0.40399296,
       0.2514121 , 0.51462509, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 1.        ]])
        prediction = Model.predict(data)
        return {
            "message": "Successful",
            'prediction': prediction[0]
        }
    except:
        return {
            'message': 'Error'
        }