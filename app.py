from flask import Flask, render_template, request, Blueprint
import pandas as pd
from sklearn.linear_model import LogisticRegression
import subprocess
from book import calculate_scaled_success_rate, categorize_bmi, predict_risk, predict_final_risk
from sklearn.model_selection import train_test_split
app = Flask(__name__)
data = pd.read_csv("child.csv")
model = LogisticRegression()
model.fit(data[['age_months', 'weight_kg']], data['label'])
app2_bp = Blueprint('app2', __name__)
@app2_bp.route('/app2', methods=['GET', 'POST'])
def app2_index():
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
app.register_blueprint(app2_bp)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    user_age = float(request.form['age'])
    user_weight = float(request.form['weight'])
    user_height = float(request.form['height'])
    df = pd.read_csv("births_by_age.csv")
    df['Lower Age'] = df['AGE'].apply(lambda x: int(x.split('-')[0]))
    df['Upper Age'] = df['AGE'].apply(lambda x: int(x.split('-')[1]))
    df['Average Age'] = (df['Lower Age'] + df['Upper Age']) / 2
    total_births = df['Count'].sum()

# Calculate success rates
    df['Success Rate'] = df['Count'] / total_births
    max_success_rate = df['Success Rate'].max()
    df['Scaled Success Rate'] = df['Success Rate'] / max_success_rate
    scaled_success_rate = calculate_scaled_success_rate(df, user_age)
    
    bmi_categories = {
        (0, 18.5): "Underweight",
        (18.5, 24.9): "Normal Weight",
        (25.0, 29.9): "Overweight",
        (30.0, 40): "Obese"
    }
    data = pd.read_csv("height_weigt.csv")
    data['bmi_category'] = data.apply(lambda row: categorize_bmi(row['weight'], row['height']), axis=1)
    X = data[['bmi_category']]
    y = data['risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_encoded = pd.get_dummies(X_train, columns=['bmi_category'])
    X_test_encoded = pd.get_dummies(X_test, columns=['bmi_category'])
    model = LogisticRegression()
    model.fit(X_train_encoded, y_train)
    user_bmi_category = categorize_bmi(user_weight, user_height)
    user_data = pd.DataFrame({'bmi_category': [user_bmi_category]})
    user_data_encoded = pd.get_dummies(user_data, columns=['bmi_category'])
    missing_cols = set(X_train_encoded.columns) - set(user_data_encoded.columns)
    for col in missing_cols:
        user_data_encoded[col] = 0
    user_data_encoded = user_data_encoded[X_train_encoded.columns]

    predicted_risk = predict_risk(user_data_encoded, model)
    final_risk_prediction = predict_final_risk(scaled_success_rate, predicted_risk)

    return render_template("index.html", user_age=user_age, user_weight=user_weight, user_height=user_height,
                           user_bmi_category=user_bmi_category, predicted_risk=final_risk_prediction)


# ... (rest of the code)
if __name__ == '__main__':
    app.run(debug=True)