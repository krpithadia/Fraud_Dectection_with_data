from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Synthetic data generation function
def generate_synthetic_data(rows):
    np.random.seed(0)
    data = {
        'amount': np.random.uniform(0, 1000, size=rows),
        'time': np.random.randint(0, 24, size=rows),
        'type': np.random.choice(['CASH-IN', 'CASH-OUT'], size=rows),
        'isFraud': np.random.choice([0, 1], p=[0.95, 0.05], size=rows)  # 5% fraudulent
    }
    return pd.DataFrame(data)

# Model training function
def train_model(df):
    X = df[['amount', 'time']]
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        rows = int(request.form.get('rows'))
        df = generate_synthetic_data(rows)
        accuracy = train_model(df)
        return render_template('index.html', accuracy=accuracy)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)