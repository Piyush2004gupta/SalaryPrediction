from flask import Flask, render_template, request
import pickle
import numpy as np

app7 = Flask(__name__)

# Load models
with open('models/scaler17.pkl', 'rb') as f:
    scaler17 = pickle.load(f)

with open('models/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('models/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('models/dbscan.pkl', 'rb') as f:
    dbscan = pickle.load(f)

with open('models/isolation_forest.pkl', 'rb') as f:
    iso = pickle.load(f)


@app7.route('/')
def home():
    return render_template('home7.html')


@app7.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    income = float(request.form['income'])
    score = float(request.form['score'])

    sample = np.array([[age, income, score]])

    scaled = scaler17.transform(sample)
    reduced = pca.transform(scaled)

    kmeans_cluster = kmeans.predict(reduced)[0]
    dbscan_cluster = dbscan.fit_predict(reduced)[0]
    anomaly = iso.predict(reduced)[0]

    anomaly_result = "Anomaly" if anomaly == -1 else "Normal"

    return render_template('home7.html',
                           kmeans_cluster=kmeans_cluster,
                           dbscan_cluster=dbscan_cluster,
                           anomaly_result=anomaly_result)


if __name__ == '__main__':
    app7.run(debug=True)
