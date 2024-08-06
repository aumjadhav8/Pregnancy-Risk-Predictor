def calculate_scaled_success_rate(df, user_age):
    scaled_success_rate = df.loc[(df['Lower Age'] <= user_age) & (df['Upper Age'] >= user_age), 'Scaled Success Rate']
    if not scaled_success_rate.empty:
        scaled_success_rate = scaled_success_rate.values[0]
        print(f"Scaled Success Rate for age {user_age}: {scaled_success_rate:.2%}")
    else:
        print("Invalid age input.")
    return scaled_success_rate

def categorize_bmi(weight, height):
    bmi_categories = {
        (0, 18.5): "Underweight",
        (18.5, 24.9): "Normal Weight",
        (25.0, 29.9): "Overweight",
        (30.0, 40): "Obese"
    }
    bmi = (weight / (height / 100) ** 2)
    for bmi_range, label in bmi_categories.items():
        if bmi_range[0] <= bmi < bmi_range[1]:
            return label
    return "Unknown"

def predict_risk(user_data_encoded, model):
    print("User Encoded Data:")
    print(user_data_encoded)

    predicted_risk = model.predict(user_data_encoded)

    if predicted_risk == 0:
        print("Low risk")
    else:
        print("High risk")
    return predicted_risk

def predict_final_risk(scaled_success_rate, high_low_risk):
    if high_low_risk == 1:  # High risk
        if scaled_success_rate >= 0.75:
            return "Moderate Risk"
        else:
            return "High Risk"
    else:  # Low risk
        if scaled_success_rate >= 0.5:
            return "Low Risk"
        elif scaled_success_rate <= 0.1:
            return "High Risk"
        else:
            return "Moderate Risk"