{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24d0941e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained XGBoost model and scaler\n",
    "with open('hr.pkl', 'rb') as model_file:\n",
    "    model = pickle.load(model_file)\n",
    "\n",
    "with open('schr.pkl', 'rb') as scaler_file:\n",
    "    scaler = pickle.load(scaler_file)\n",
    "\n",
    "# Create the web app\n",
    "st.title('Salary Prediction App')\n",
    "\n",
    "# Input fields\n",
    "age = st.number_input('Age', min_value=0, max_value=120, value=30)\n",
    "education = st.selectbox('Education Level', ['High School or less', 'Intermediate', 'Graduation', 'PG'])\n",
    "experience_months = st.number_input('Months of Experience', min_value=0, max_value=600, value=60)  # Assuming max experience is 50 years\n",
    "\n",
    "# Convert education to numeric encoding\n",
    "education_mapping = {'High School or less': 0, 'Intermediate': 1, 'Graduation': 2, 'PG': 3}\n",
    "education_encoded = education_mapping[education]\n",
    "\n",
    "# Prepare the feature vector\n",
    "features = np.array([[age, education_encoded, experience_months]], dtype=np.float64)\n",
    "\n",
    "# Scale the features\n",
    "features_scaled = scaler.transform(features)\n",
    "\n",
    "# Predict salary\n",
    "predicted_salary = model.predict(features_scaled)\n",
    "\n",
    "# Display the result\n",
    "st.write(f\"Predicted Salary: ${predicted_salary[0]:,.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
