from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('iris_model.pkl')  # Make sure 'iris_model.pkl' is in your project directory

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
