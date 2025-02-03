from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Converting categorical values into numerical to prepare for modeling

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each categorical variable
for col in categorical_vars:
  hotel_bookings_df[col] = label_encoder.fit_transform(hotel_bookings_df[col])

# Display the updated DataFrame with numerical encoded categorical variables
print(hotel_bookings_df.head())


# 3. Modeling Stage


# Logistic Regression

# Define features (X) and target variable (y)
X = hotel_bookings_df.drop(columns=['is_canceled'])
y = hotel_bookings_df['is_canceled']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
logreg_model = LogisticRegression(max_iter=1000)

# Train the model on the training data
logreg_model.fit(X_train, y_train)

# Predictions on the testing data
y_pred = logreg_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Increase the number of iterations for the solver

logreg_model = LogisticRegression(max_iter=10000)

# Train the model on the training data
logreg_model.fit(X_train, y_train)

# Predictions on the testing data
y_pred = logreg_model.predict(X_test)

# Second Model evaluation
# Accuracy, Precision, Recall, and F1-score for logistic regression
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy (after increasing max_iter):", accuracy)
print("\nPrecision (after increasing max_iter):", precision)
print("Recall (after increasing max_iter):", recall)
print("F1-score (after increasing max_iter):", f1)

# Classification report
print("\nClassification Report (after increasing max_iter):")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix (after increasing max_iter):")
print(confusion_matrix(y_test, y_pred))


# Normalization technique

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize logistic regression model with scaled data
logreg_model_scaled = LogisticRegression(max_iter=10000)

# Train the model on the scaled training data
logreg_model_scaled.fit(X_train_scaled, y_train)

# Predictions on the scaled testing data
y_pred_scaled = logreg_model_scaled.predict(X_test_scaled)

# Model evaluation with scaled data (Third Model)

# Accuracy, Precision, Recall, and F1-score for logistic regression after scaling data
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
precision_scaled = precision_score(y_test, y_pred_scaled)
recall_scaled = recall_score(y_test, y_pred_scaled)
f1_scaled = f1_score(y_test, y_pred_scaled)

print("\nAccuracy (after scaling data):", accuracy_scaled)
print("\nPrecision (after scaling data):", precision_scaled)
print("Recall (after scaling data):", recall_scaled)
print("F1-score (after scaling data):", f1_scaled)

# Classification report with scaled data
print("\nClassification Report (after scaling data):")
print(classification_report(y_test, y_pred_scaled))

# Confusion matrix with scaled data
print("\nConfusion Matrix (after scaling data):")
print(confusion_matrix(y_test, y_pred_scaled))


# Feature Importance Ranking

# Extract coefficients from the logistic regression model with scaled data
coefficients_scaled = logreg_model_scaled.coef_

# Get feature names
feature_names_scaled = X.columns

# Create a dictionary to store feature coefficients
feature_coefficients_scaled = dict(zip(feature_names_scaled, coefficients_scaled[0]))

# Sort the features based on their coefficients
sorted_features_scaled = sorted(feature_coefficients_scaled.items(), key=lambda x: abs(x[1]), reverse=True)

# Display the top features based on their coefficients
top_features_scaled = sorted_features_scaled[:10]  # Get the top 10 features
for feature, coefficient in top_features_scaled:
    print(f"Feature: {feature}, Coefficient: {coefficient}")

top_features_names_scaled = [feature[0] for feature in top_features_scaled]
top_features_coefficients_scaled = [feature[1] for feature in top_features_scaled]

plt.figure(figsize=(10, 6))
plt.barh(top_features_names_scaled, top_features_coefficients_scaled)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Top Features by Coefficients (Scaled Model)')
plt.gca().invert_yaxis()  # Invert y-axis to display the highest coefficient at the top
plt.show()


# Define the columns to keep based on feature importance analysis
columns_to_keep = [
    'arrival_date_year', 'reservation_status_date', 'arrival_date_week_number',
    'reservation_status', 'required_car_parking_spaces', 'country',
    'stays_in_week_nights', 'hotel', 'assigned_room_type', 'total_of_special_requests'
]

# Filter the DataFrame to keep only the relevant columns
X = hotel_bookings_df[columns_to_keep]
y = hotel_bookings_df['is_canceled']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize logistic regression model with scaled data
logreg_model_scaled = LogisticRegression(max_iter=10000)

# Train the model on the scaled training data
logreg_model_scaled.fit(X_train_scaled, y_train)

# Predictions on the scaled testing data
y_pred_scaled = logreg_model_scaled.predict(X_test_scaled)

# Model evaluation with scaled data (Third Model)
# Accuracy, Precision, Recall, and F1-score for logistic regression after scaling data
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
precision_scaled = precision_score(y_test, y_pred_scaled)
recall_scaled = recall_score(y_test, y_pred_scaled)
f1_scaled = f1_score(y_test, y_pred_scaled)

print("\nAccuracy (after scaling data):", accuracy_scaled)
print("\nPrecision (after scaling data):", precision_scaled)
print("Recall (after scaling data):", recall_scaled)
print("F1-score (after scaling data):", f1_scaled)

# Classification report with scaled data
print("\nClassification Report (after scaling data):")
print(classification_report(y_test, y_pred_scaled))

# Confusion matrix with scaled data
print("\nConfusion Matrix (after scaling data):")
print(confusion_matrix(y_test, y_pred_scaled))
