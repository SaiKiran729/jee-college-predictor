import os
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = pickle.load(open("model1.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            institute_type = request.form.get('institute_type')
            jee_rank = int(request.form.get('jee_rank') or 0)
            opening_rank = request.form.get('opening_rank')
            closing_rank = request.form.get('closing_rank')

            if not opening_rank or not closing_rank:
                buffer = int(jee_rank * 0.05)
                opening_rank = max(1, jee_rank - buffer)
                closing_rank = jee_rank + buffer
            else:
                opening_rank = int(opening_rank)
                closing_rank = int(closing_rank)

            category = request.form.get('category')
            gender_pool = request.form.get('gender_pool')
            branch = request.form.get('branch')

            institute_type_encoded = 0 if institute_type.lower() == 'iit' else 1
            pool_encoded = 0 if gender_pool.lower() == 'male' else 1

            category_mapping = {
                'GEN': 0, 'OBC-NCL': 1, 'SC': 2, 'ST': 3
            }
            category_encoded = category_mapping.get(category, -1)
            if category_encoded == -1:
                return render_template('error.html', message="Invalid category input.")

            if 1 <= opening_rank <= 500:
                n_neighbors = 16
            elif 500 < opening_rank <= 2000:
                n_neighbors = 12
            elif 2000 < opening_rank <= 10000:
                n_neighbors = 10
            elif 10000 < opening_rank <= 50000:
                n_neighbors = 8
            elif 50000 < opening_rank <= 70000:
                n_neighbors = 5
            else:
                n_neighbors = 3

            input_data = pd.DataFrame({
                'category': [category_encoded],
                'pool': [pool_encoded],
                'institute_type': [institute_type_encoded],
                'opening_rank': [opening_rank],
                'closing_rank': [closing_rank]
            })

            nearest_neighbors_indices = model.kneighbors(input_data, n_neighbors=n_neighbors)[1][0]
            y1 = pd.read_csv("data.csv")
            y1_reset = y1.reset_index(drop=True)
            predicted_colleges = y1_reset.iloc[nearest_neighbors_indices][['institute_short', 'program_name', 'degree_short']]

            if institute_type.lower() == "nit":
                predicted_colleges = predicted_colleges[predicted_colleges['institute_short'].str.contains("NIT", case=False)]
            elif institute_type.lower() == "iit":
                predicted_colleges = predicted_colleges[predicted_colleges['institute_short'].str.contains("IIT|NIT", case=False)]

            branch_keywords = {
                "CSE": "Computer Science and Engineering",
                "IT": "Information Technology",
                "ECE": "Electronics and Communication Engineering",
                "EEE": "Electrical Engineering",
                "MECH": "Mechanical Engineering",
                "CIVIL": "Civil Engineering",
                "Chemical" : "Chemical Engineering",
                "Artifical Intelligence":"Artificial Intelligence",
                "Artifical Intelligence & Data Science":"Artificial Intelligence and Data Science",
                "Aeronautical" : "Aeronautical Engineering"
            }

            if branch != "All":
                keyword = branch_keywords.get(branch, branch).lower()
                predicted_colleges = predicted_colleges[
                    predicted_colleges['program_name'].str.lower().str.contains(keyword, na=False)
                ]

            colleges_list = predicted_colleges.to_dict('records')
            return render_template('result.html', colleges=colleges_list)

        except Exception as e:
            return render_template('error.html', message=str(e))

    return render_template('form.html')

@app.route('/cutoff_trends')
def cutoff_trends():
    college = request.args.get('college')
    branch = request.args.get('branch')

    df = pd.read_csv("data.csv")

    filtered = df[
        (df['institute_short'].str.lower() == college.lower()) &
        (df['program_name'].str.lower().str.contains(branch.lower(), na=False))
    ]

    if 'year' not in filtered.columns or filtered.empty:
        return jsonify({'years': [], 'ranks': []})

    trends = filtered.groupby('year')['closing_rank'].mean().reset_index().sort_values('year')

    return jsonify({
        'years': trends['year'].tolist(),
        'ranks': trends['closing_rank'].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
