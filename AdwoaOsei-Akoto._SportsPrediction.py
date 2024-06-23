#Importing necessary libraries and files
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

#QUESTION 1
#Process the data
def preprocess_data(file_path, columns_to_drop):
    # Read the CSV file
    data = pd.read_csv(file_path, low_memory = False)
    
    # Drop specified columns
    data = data.drop(columns=columns_to_drop, axis=1)
    
    # Drop columns with 30% or more null values
    biased_threshold = 0.30 * len(data)
    data = data.loc[:, data.isna().sum() < biased_threshold]
    
    # Separate numeric data from non-numeric data
    numeric_data = data.select_dtypes(include=np.number)
    non_numeric_data = data.select_dtypes(include=['object'])
    
    # Impute missing values for numeric data with mean
    numeric_data = numeric_data.apply(lambda col: col.fillna(col.mean()))
    
    # Impute missing values for non-numeric data with mode
    for column in non_numeric_data.columns:
        mode = non_numeric_data[column].mode()[0]
        non_numeric_data[column].fillna(mode, inplace=True)
    
    # Encode categorical columns
    # Assuming non_numeric_data is a DataFrame containing non-numeric columns
    label_encoded_data = non_numeric_data.copy()

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Iterate through each column and apply LabelEncoder
    for column in label_encoded_data.columns:
        label_encoded_data[column] = le.fit_transform(label_encoded_data[column])

    
    # Combine numeric data and one-hot encoded data
    processed_data = pd.concat([numeric_data, label_encoded_data], axis=1)

    processed_data = pd.DataFrame(processed_data)
    
    return processed_data

# Run file into function:
file_path = r"C:\Users\Dell Inspiron\Documents\School_2024_2\Intro to AI\male_players (legacy).csv"
columns_to_drop = [
    'player_id', 'dob', 'player_tags', 'club_contract_valid_until_year', 'player_url', 'club_jersey_number','fifa_update',
    'long_name', 'short_name', 'league_id', 'player_face_url', 'nationality_id','preferred_foot','club_contract_valid_until_year',
    'fifa_update_date', 'club_position', 'league_name', 'club_team_id', 'nation_team_id', 'player_traits','club_joined_date','league_level',
    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'ldm', 'cdm', 'rdm',  'lb', 'cb','nationality_name','real_face',
    'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lwb', 'rwb', 'lcb', 'rcb', 'rb', 'gk', 'player_face_url', 'ls', 'st','body_type','fifa_version'
]
processed_data = preprocess_data(file_path, columns_to_drop)
processed_data.head()

#Check number of missing values
processed_data.isnull().sum()

#QUESTION 2: Getting the right feauture subsets
#Feature Importance using RandomForest.
#Pick the strongly correlated variables from the lot
# Split the data into training and testing sets
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(processed_data.drop('overall', axis=1), processed_data['overall'], test_size=0.2, random_state=0)

# Create a Random Forest Regressor with fewer estimators
forest = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)

# Fit the model on the training data
forest.fit(X_train, y_train)

# Extract feature names
feature_names = X_train.columns

# Print feature importances
#for name, score in zip(feature_names, forest.feature_importances_):
 #   print(name, score)


# Extract feature importances and sort them
feature_importances = forest.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top 13 features 
top_N = 13 
top_features = importance_df['Feature'].head(top_N).tolist()
top_values = importance_df['Importance'].head(top_N).tolist()

print(f"Top features selected: {top_features} and their respective values are {top_values}.")

#Creating new dataframe with just strongly correlated features
new_x = processed_data[top_features]

#Scaling the independent variables
X = scaler.fit_transform(new_x)

 #QUESTION 3
## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, processed_data['overall'], test_size=0.2, random_state=0)

#Using Random Forest Regressor to train and predict the values
model_rf=RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf =model_rf.predict(X_test)
#print(y_pred_rf)

#Testing the accuracy of the Random Forest Prediction
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("The mean absolute error without cross validation is : ", mae_rf)

#Cross-validation for the RandomTreeRegressor method to improve the data predictions
kf = KFold(n_splits = 5,shuffle=True, random_state=42)
scores = cross_val_score(model_rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
scores = abs(scores)
#print("Cross-validation Scores (MAE):", scores)
print("The average error with cross-validation is", scores.mean())

#Using exGradientBooster(xgbooster) to train datasets
model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, objective='reg:squarederror')
model.fit(X_train, y_train)
y_pred_xg = model.predict(X_test)

mae_xg = mean_absolute_error(y_test, y_pred_xg)
print("The mean absolute error without cross validation with an xgbooster is :",mae_xg)

#Cross-validation for the xgboost method to improve the data predictions
kf = KFold(n_splits = 5,shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
scores = abs(scores)
#print("Cross-validation Scores (MAE):", scores)
print('The mean absolute error cross validation with an xgbooster is :', scores.mean())

#Using GradientBooster to train datasets
model_g = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, loss='squared_error')
model_g.fit(X_train, y_train)
y_pred_g = model_g.predict(X_test)

mae_g = mean_absolute_error(y_test, y_pred_g)
print("The mean absolute error without cross validation with a gradient booster is :",mae_g)

#Cross-validation for the xgboost method to improve the data predictions
kf = KFold(n_splits = 5,shuffle=True, random_state=42)
scores = cross_val_score(model_g, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
scores = abs(scores)
#print("Cross-validation Scores (MAE):", scores)
print('The mean absolute error cross validation with a gradient booster is :', scores.mean())

# #QUESTION 4
#Fine tuning models to get a lower MAE value
#RandomForestRegressorModel since it showed the lowest MAE
cv = KFold(n_splits = 3)
grid_hp = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40, 50]
}
grid_search = GridSearchCV( estimator = model_rf, param_grid = grid_hp, scoring = 'neg_mean_absolute_error', cv = cv, n_jobs = -1, verbose = 0)

#Fitting grid_search onto the data
grid_search.fit(X_train, y_train)

# Best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
print("Mean Absolute Error (MAE) with the best model:", mae_best)

#Retraining and retesting the model with cross validation
kf = KFold(n_splits = 5,shuffle=True, random_state=42)
scores = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
scores = abs(scores)
#print("Cross-validation Scores (MAE):", scores)
print('The mean absolute error cross validation with the best model is :', scores.mean())

#Retraing and retesting the model without cross validation
best_model.fit(X_train, y_train)
y_pred_b = best_model.predict(X_test)
mae_b = mean_absolute_error(y_test, y_pred_b)
print("The mean absolute error without cross validation is :",mae_b)

# #QUESTION 5

#Testing model with new dataset
#Processing testing data
file_path = r"C:\Users\Dell Inspiron\Documents\School_2024_2\Intro to AI\players_22-1.csv"
columns_to_drop = [
    'sofifa_id', 'dob', 'player_tags', 'club_contract_valid_until', 'player_url', 'club_jersey_number',
    'long_name', 'short_name', 'player_face_url', 'nationality_id','preferred_foot',
     'club_position', 'league_name', 'club_team_id', 'nation_team_id', 'player_traits','club_joined','league_level',
    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'ldm', 'cdm', 'rdm',  'lb', 'cb','nationality_name','real_face',
    'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lwb', 'rwb', 'lcb', 'rcb', 'rb', 'gk', 'player_face_url', 'ls', 'st','body_type','nation_jersey_number',
    'nation_position', 'club_logo_url','club_flag_url', 'nation_logo_url','nation_flag_url'
]

testing_data = preprocess_data(file_path, columns_to_drop)
testing_data.head()

## Split the data into training and testing sets
test_X = testing_data[top_features]
test_X = scaler.fit_transform(test_X)
newX_train, newX_test, newy_train, newy_test = train_test_split(test_X, testing_data['overall'], test_size=0.2, random_state=0)

#Testing the performance of the trained model training new data with MAE metrics
best_model.fit(newX_train, newy_train)
test_pred = best_model.predict(newX_test)
mae_test = mean_absolute_error(newy_test, test_pred)
print(f'This is the MAE value for the testing the model with the new data: {mae_test}. Therefore showing relatively how good the model is as its prediction varies slightly from the actual value.')

# #QUESTION 6
#Employing the model on a website
#Saving the model
with open(r'C:\Users\Dell Inspiron\Documents\School_2024_2\Intro to AI.pkl', 'wb') as file:
    pkl.dump(best_model, file)

rmse = mean_squared_error(newy_test, test_pred, squared=False)
rmse

# Find the average of the overall feature and use that as a true value/target value for finding the confidence level
processed_data['overall'].mean()

#Saving scaler as module to scale input to model
with open(r'C:\Users\Dell Inspiron\Documents\School_2024_2\Intro to AI.pkl', 'wb') as file:
    pkl.dump(scaler, file)




