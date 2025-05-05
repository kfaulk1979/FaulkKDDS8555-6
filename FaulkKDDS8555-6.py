import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report as cr, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier


train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Multi-class Prediction of Obesity Risk/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Multi-class Prediction of Obesity Risk/test.csv")

# Encode categorical variables
label_encoders = {}
for col in train.select_dtypes(include=['object']).columns:
    if col != 'NObeyesdad':  # Leave the target for later encoding
        # Combine train and test values for this column
        combined = pd.concat([train[col], test[col]], axis=0)
        le = LabelEncoder().fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        label_encoders[col] = le


# Encode the target variable
target_le = LabelEncoder()
train['NObeyesdad'] = target_le.fit_transform(train['NObeyesdad'])

# Define features and target
X = train.drop(columns=['id', 'NObeyesdad'])  # Drop id and target
y = train['NObeyesdad']

# Ensure test data has the same feature columns
X_test = test[X.columns]

# Train/test split for proper validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=1)
dt_scores = cross_val_score(dt_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_val)

# Classification report
print("Decision Tree Classification Report:")
print(cr(y_val, dt_preds))

# Feature importance
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=target_le.classes_, filled=True)
plt.title("Decision Tree Structure")
plt.show()

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_val, dt_preds, display_labels=target_le.classes_, cmap='Blues', xticks_rotation=45)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Bagging Classifier
bag_model = BaggingClassifier(n_estimators=30, random_state=1)
bag_scores = cross_val_score(bag_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
bag_model.fit(X_train, y_train)
bag_preds = bag_model.predict(X_val)

# Classification report
print("Bagging Classifier Classification Report:")
print(cr(y_val, bag_preds))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_val, bag_preds, display_labels=target_le.classes_, cmap='Blues', xticks_rotation=45)
plt.title("Bagging Confusion Matrix")
plt.show()


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_scores = cross_val_score(rf_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

# Classification report
print("Random Forest Classification Report:")
print(cr(y_val, rf_preds))

# Feature importance
rf_feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_feat_importance.values, y=rf_feat_importance.index)
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_val, rf_preds, display_labels=target_le.classes_, cmap='Blues', xticks_rotation=45)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=1)
gb_scores = cross_val_score(gb_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_val)

# Classification report
print("Gradient Boosting Classification Report:")
print(cr(y_val, gb_preds))

# Feature importance
gb_feat_importance = pd.Series(gb_model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=gb_feat_importance.values, y=gb_feat_importance.index)
plt.title("Top 10 Gradient Boosting Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_val, gb_preds, display_labels=target_le.classes_, cmap='Blues', xticks_rotation=45)
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

# Prepare model comparison
comparison_df = pd.DataFrame({
    "Model": ["Decision Tree", "Bagging", "Random Forest", "Gradient Boosting"],
    "CV Accuracy (Train Set)": [
        np.mean(dt_scores),
        np.mean(bag_scores),
        np.mean(rf_scores),
        np.mean(gb_scores)
    ],
    "Validation Accuracy": [
        dt_model.score(X_val, y_val),
        bag_model.score(X_val, y_val),
        rf_model.score(X_val, y_val),
        gb_model.score(X_val, y_val)
    ]
})

print("\nModel Comparison Summary:")
print(comparison_df)
comparison_df.to_csv("model_comparison_summary.csv", index=False)

# Prepare test data
def save_submission(preds, filename):
    preds_labels = target_le.inverse_transform(preds)
    submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': preds_labels})
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


# Predict and save
save_submission(dt_model.predict(X_test), 'decision_tree_submission.csv')
save_submission(bag_model.predict(X_test), 'bagging_submission.csv')
save_submission(rf_model.predict(X_test), 'random_forest_submission.csv')
save_submission(gb_model.predict(X_test), 'gradient_boosting_submission.csv')