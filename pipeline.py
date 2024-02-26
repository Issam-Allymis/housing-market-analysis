from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from feature_engine.selection import SmartCorrelatedSelection

import numpy as np
import pandas as pd
df = (pd.read_csv("outputs/datasets/collection/house-price-2021.csv")
      .drop(labels=['WoodDeckSF', 'EnclosedPorch'], axis=1)  
                    
  )

print(df.shape)
df.info()#head()from feature_engine.selection import SmartCorrelatedSelection

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale numerical features
    ('correlation_selection', SmartCorrelatedSelection(variables=None, method="pearson", threshold=0.9, selection_method="variance")),  # Select correlated features
    ('model', RandomForestRegressor(random_state=101))  # Add your model
])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['SalePrice'], axis=1),
    df['SalePrice'],
    test_size=0.2,
    random_state=101,
)
# Now you can fit and transform your data with the pipeline
pipeline.fit(X_train, y_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train.shape)


pipeline_data_cleaning_feat_eng = PipelineDataCleaningAndFeatureEngineering()
X_train = pipeline_data_cleaning_feat_eng.fit_transform(X_train, y_train) 
X_test = pipeline_data_cleaning_feat_eng.transform(X_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)