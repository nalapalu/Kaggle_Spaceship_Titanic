import sys
import joblib
import pandas as pd


sys.path.append("..")

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

test = pd.read_csv("../../data/processed/test.csv")
# test = test.drop(test.columns[1], axis=1)

# --------------------------------------------------------------
# Load model
# --------------------------------------------------------------

model, ref_cols, target = joblib.load("../../models/model.pkl")

# --------------------------------------------------------------
# Make predictions
# --------------------------------------------------------------

X_new = test[ref_cols]
predictions = model.predict(X_new)

test_results = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})


# accuracy = accuracy_score(test['Survived'], predictions)
# print("Accuracy:", accuracy)


test_results.to_csv('../../data/processed/chiran_submission.csv', index=False)



# --------------------------------------------------------------
# Evaluate results
# --------------------------------------------------------------

# rmse = np.sqrt(mean_squared_error(y_new, predictions))
# r2 = r2_score(y_new, predictions)
# accuracy = accuracy_score(y_new, predictions)

# print("Accuracy:", accuracy)
# print("RMSE:", rmse)
# print("R2:", r2)

# # Visualize results
# plot_predicted_vs_true(y_new, predictions)