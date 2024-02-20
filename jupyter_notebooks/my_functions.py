from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def pipeline_randomforest_reg():
        pipeline = Pipeline([
                ('mean_median_imputer', MeanMedianImputer(imputation_method='median')),
                ('categorical_imputer', CategoricalImputer(imputation_method='missing')),
                ('feat_scaling', StandardScaler()),
                ('model', RandomForestRegressor())
        ])

        return pipeline





