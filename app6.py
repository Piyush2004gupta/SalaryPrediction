from flask import Flask, render_template, request
import pickle
import numpy as np

app6=Flask(__name__)

# Load model and scaler
model = pickle.load(open('models/salary_model.pkl', 'rb'))
scaler15 = pickle.load(open('models/scaler15.pkl', 'rb'))

@app6.route('/')
def home():
    return render_template('home6.html')

@app6.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    gender = request.form['gender']
    education = request.form['education']
    job_title = request.form['job_title']
    experience = float(request.form['experience'])

    gender_encoded = 1 if gender.lower() == 'male' else 0
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    education_encoded = education_mapping.get(education, 0)

    job_mapping = {
        'Software Engineer': 0, 'Data Analyst': 1, 'Senior Manager': 2, 'Sales Associate': 3, 'Director': 4,
        'Marketing Analyst': 5, 'Product Manager': 6, 'Sales Manager': 7, 'Marketing Coordinator': 8,
        'Senior Scientist': 9, 'Software Developer': 10, 'HR Manager': 11, 'Financial Analyst': 12,
        'Project Manager': 13, 'Customer Service Rep': 14, 'Operations Manager': 15, 'Marketing Manager': 16,
        'Senior Engineer': 17, 'Data Entry Clerk': 18, 'Sales Director': 19, 'Business Analyst': 20,
        'VP of Operations': 21, 'IT Support': 22, 'Recruiter': 23, 'Financial Manager': 24
    }

    job_encoded = job_mapping.get(job_title, 0)
    final_input = np.array([[age, gender_encoded, education_encoded, job_encoded, experience]])
    final_scaled = scaler15.transform(final_input)
    prediction = model.predict(final_scaled)[0]

    return render_template('home6.html', prediction_text=f"Predicted Salary: ₹{prediction:,.2f}")

if __name__ == '__main__':
    app6.run(debug=True)
