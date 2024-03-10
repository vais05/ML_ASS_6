import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # Consider both encoders for flexibility
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(data_path, target_column):
 
    try:
        data = pd.read_csv("C:\Users/vaish\OneDrive - Amrita vishwa vidyapeetham\Documents/4th SEM\ML\ml_dataset.csv")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None 
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    if categorical_cols:
        if len(set(data[categorical_cols[0]].unique())) > 2:  
            encoder = OneHotEncoder(sparse=False)
        else:
            encoder = LabelEncoder()
        encoded_data = pd.concat([data[numerical_cols] for numerical_cols in data.columns if numerical_cols not in categorical_cols],
                                encoder.fit_transform(data[categorical_cols]), axis=1)
    else:
        encoded_data = data

    # Separate features and target variable
    X = encoded_data.drop(target_column, axis=1)
    y = encoded_data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize/standardize features (if necessary)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder if categorical_cols else None, scaler

def train_and_evaluate_mlp(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,), solver='lbfgs'):


    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, random_state=42)
    mlp.fit(X_train, y_train)

    