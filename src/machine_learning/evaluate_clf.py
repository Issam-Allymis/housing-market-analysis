import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# code copied from "Modeling and Evaluation" notebooks


"""def confusion_matrix_and_report(X, y, pipeline, label_map):

    prediction = pipeline.predict(X)

    st.write('#### Confusion Matrix')
    st.code(pd.DataFrame(confusion_matrix(y_true=prediction, y_pred=y),
                         columns=[["Actual " + sub for sub in label_map]],
                         index=[["Prediction " + sub for sub in label_map]]
                         ))

    st.write('#### Classification Report')
    st.code(classification_report(y, prediction, target_names=label_map), "\n")


# code copied from "Modeling and Evaluation" notebooks
def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    st.info("Train Set")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.info("Test Set")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)"""


def reg_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("#### Train Set ####")
    y_train_pred = pipeline.predict(X_train)
    st.write("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
    st.write("R2 Score (Train):", r2_score(y_train, y_train_pred))

    st.write("\n#### Test Set ####")
    y_test_pred = pipeline.predict(X_test)
    st.write("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
    st.write("R2 Score (Test):", r2_score(y_test, y_test_pred))

