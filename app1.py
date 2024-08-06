from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the CSV dataset
data = pd.read_csv("child.csv")

# Initialize the logistic regression model
model = LogisticRegression()
model.fit(data[['age_months', 'weight_kg']], data['label'])

@app.route('/', methods=['GET', 'POST'])
def app1_index():
    prediction = None
    input_age = None
    input_weight = None
    if request.method == 'POST':
        input_age = float(request.form['age_months'])
        input_weight = float(request.form['weight_kg'])
        age_months = input_age
        weight_kg = input_weight
        user_input = pd.DataFrame({'age_months': [age_months], 'weight_kg': [weight_kg]})
        proba = model.predict_proba(user_input)
        custom_threshold = 0.6
        prediction = (proba[:, 1] >= custom_threshold).astype(int)[0]
    return render_template('index1.html', prediction=prediction, input_age=input_age, input_weight=input_weight)


if __name__ == '__main__':
    app.run(debug=True)