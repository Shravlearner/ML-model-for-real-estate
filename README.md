Bangalore House Price Prediction

An end-to-end machine learning project to predict house prices in Bangalore based on various features like location, square footage, number of bedrooms, and more. The project demonstrates the entire data science workflow from data preprocessing to model deployment.

Table of Contents

1. Project Overview
2. Dataset
3. Dependencies
4. Project Structure
5. Key Features
6. How to Run the Project
7. Results
8. Future Improvements
9. Acknowledgments

Project Overview

The objective of this project is to build a machine learning model that predicts house prices in Bangalore. The project focuses on the following:

- Data cleaning and preprocessing.
- Feature engineering.
- Model training and evaluation.
- Deploying the model as a web application using Flask.

Dataset

- Source: The dataset is sourced from Kaggle and contains information on houses in Bangalore.
- Features:
  - Location
  - Square footage
  - Number of bedrooms
  - Number of bathrooms
  - Price (target variable)

Dependencies

The following Python libraries are required to run this project:

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Flask

Install the dependencies using the following command:

pip install -r requirements.txt

Project Structure

bhp/
|
├── Client/
      |-app.css
      |-app.html
      |-app.js
├── Model/  
      |-banglore_home_prices_final (2).ipynb
      |-Bengaluru_House_Data.csv
      |-bhp.csv
     
├── Server/   
      |-server.py
      |-util.py
      |-artifacts
            |-banglore_home_prices_final.pickle
            |-columns.json
├── requirements.txt    
├── README.md           

Key Features

1. Data Preprocessing:
   - Handled missing values.
   - Removed outliers.
   - Encoded categorical variables.
   - Performed feature scaling.

2. Modeling:
   - Trained multiple regression models (Linear Regression, Decision Trees, etc.).
   - Performed hyperparameter tuning.
   - Selected the best-performing model based on evaluation metrics like RMSE and R-squared.

3. Web Application:
   - Integrated the model into a Flask web application.
   - User-friendly interface to input house features and predict prices.


How to Run the Project

1. Clone the repository:


   git clone https://github.com/codebasics/bangalore_price_prediction.git
   cd bangalore_price_prediction
   

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

4. Open the application in your browser at `http://127.0.0.1:5000/`.

---

## Results

- The best-performing model achieved an R-squared value of **0.85** on the test dataset.
- The web application can predict house prices within a reasonable margin of error based on user inputs.

---

## Future Improvements

1. Expand the dataset to include more features like proximity to schools, hospitals, and public transportation.
2. Experiment with advanced machine learning algorithms like Gradient Boosting and Neural Networks.
3. Deploy the application to a cloud platform like AWS, GCP, or Heroku for wider accessibility.

---

## Acknowledgments

This project was inspired by the Codebasics YouTube tutorial on machine learning. Special thanks to Codebasics for providing guidance and insights into building end-to-end machine learning projects.
