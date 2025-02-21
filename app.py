# Importing Flask and other modules
import os
from flask import Flask, request, render_template
import pickle
import pandas as pd
from collections.abc import Mapping


# Flask constructor
app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.urandom(24)  

# Load the trained model
model = pickle.load(open("model1.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            # Get form data
            institute_type = request.form.get('institute_type')
            opening_rank = int(request.form.get('opening_rank'))
            closing_rank = int(request.form.get('closing_rank'))
            round_no = int(7)
            category = request.form.get('category')
            quota = 'AI'
            gender_pool = request.form.get('gender_pool')

            # Encode categorical variables
            institute_type_encoded = 0 if institute_type.lower() == 'iit' else 1
            pool_encoded = 0 if gender_pool.lower() == 'male' else 1

            # Map category and quota (ensure these match the model's training encoding)
            category_mapping = {
                'GEN': 0, 'OBC-NCL': 1, 'SC': 2, 'ST': 3
            }
            quota_mapping = {'AI': 0}

            category_encoded = category_mapping.get(category, -1)
            quota_encoded = quota_mapping.get(quota, -1)

            # Handle invalid inputs
            if category_encoded == -1 or quota_encoded == -1:
                return render_template('error.html', message="Invalid category or quota input.")

            # Determine the number of colleges to display based on opening rank
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

            # Create a DataFrame from the input data
            input_data = pd.DataFrame({
                'category': [category_encoded],
                'quota': [quota_encoded],
                'pool': [pool_encoded],  
                'institute_type': [institute_type_encoded],
                'round_no': [round_no],
                'opening_rank': [opening_rank],
                'closing_rank': [closing_rank]
            })

            # Use the model to predict the colleges
            nearest_neighbors_indices = model.kneighbors(input_data, n_neighbors=n_neighbors)[1][0]
            y1 = pd.read_csv("data.csv")
            y1_reset = y1.reset_index(drop=True)  # Reset index to align with kneighbors output

            # Fetch the corresponding colleges
            predicted_colleges = y1_reset.iloc[nearest_neighbors_indices][['institute_short', 'program_name', 'degree_short']]

            # Apply filter based on user selection
            if institute_type.lower() == "nit":
                predicted_colleges = predicted_colleges[predicted_colleges['institute_short'].str.contains("NIT", case=False)]
            elif institute_type.lower() == "iit":
                predicted_colleges = predicted_colleges[predicted_colleges['institute_short'].str.contains("IIT|NIT", case=False)]

            # Convert the DataFrame to a list of dictionaries for rendering
            colleges_list = predicted_colleges.to_dict('records')

            return render_template('result.html', colleges=colleges_list)

        except Exception as e:
            return render_template('error.html', message=str(e))

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
