import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_telco_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_saleprice_study_body(): 

    # load data
    df = load_telco_data()

    
    vars_to_study = ['YearBuilt', 'BsmtFinSF1',
                     'BsmtUnfSF', 'TotalBsmtSF', 'OverallQual']

    st.write("### SalePrice Study")
    st.info(
        f"* The client is interested in understanding the patterns within the housing market data "
        f"to identify the most relevant variables "
        f"correlated with high sale prices.")

    # inspect data
    if st.checkbox("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted to better understand how  "
        f"the variables are correlated with Sale Price levels.\n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "02 - Churned Customer Study" notebook - "Conclusions and Next steps" section
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* A higher saleprice is typically associated with the First Floor square feet  having a larger surface area **['1stFlrSF']**. \n"
        f"* A higher saleprice is typically associated with a larger Size of garage in square feet **['GarageArea']**. \n"
        f"* A higher saleprice is typically associated with a larger Above grade (ground) living area in square feet **['GrLivArea']**. \n"
        f"* A higher saleprice is typically associated with a higher Rate of the overall material/quality and finish of the house **['OverallQual']**. \n"
        f"* A higher saleprice is typically associated with the remodel date **['YearRemodAdd']**. \n"
    )

    # Code copied from "02 - Churned Customer Study" notebook - "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("SalePrice Levels per Variable"):
        saleprice_level_per_variable(df_eda)

    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.write(
            f"* Information in navy indicates the profile of properties with higher \n"
            f"Sale Price levels, while light blue (almost white) represents properties \n"
            f"with lower Sale Price levels.")
        parallel_plot_saleprice(df_eda)


# function created using "02 - Churned Customer Study" notebook code - "Variables Distribution by Churn" section
def saleprice_level_per_variable(df_eda):
    target_var = 'SalePrice'

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


# code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
def plot_categorical(df, col, target_var, max_labels=5):
    fig, axes = plt.subplots(figsize=(12, 5))
    # Select the top max_labels categories based on their counts
    top_categories = df[col].value_counts().head(max_labels).index
    sns.countplot(data=df[df[col].isin(top_categories)], x=col, hue=target_var,
                  order=top_categories)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
     # Limit the number of legend labels
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[:max_labels], labels[:max_labels], title=target_var)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
def plot_numerical(df, col, target_var, max_labels=5):
    fig, axes = plt.subplots(figsize=(8, 5))
    # Select the top max_labels categories based on their counts
    top_categories = df[col].value_counts().head(max_labels).index
    data_to_plot = df[df[col].isin(top_categories)]
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")

    legend_labels = data_to_plot[target_var].unique()[:max_labels]
    plt.legend(labels=legend_labels)

    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# function created using "02 - Churned Customer Study" notebook code - Parallel Plot section
def parallel_plot_saleprice(df_eda):

    # hard coded from "disc.binner_dict_['tenure']"" result,
    OverallQual_map = [-np.inf, 5.0, 6.0, 7.0, np.inf]
    # found at "02 - Churned Customer Study" notebook
    # under "Parallel Plot" section
    disc = ArbitraryDiscretiser(binning_dict={'OverallQual': OverallQual_map})
    df_parallel = disc.fit_transform(df_eda)

    n_classes = len(OverallQual_map) - 1
    classes_ranges = disc.binner_dict_['OverallQual'][1:-1]
    LabelsMap = {}
    for n in range(0, n_classes):
        if n == 0:
            LabelsMap[n] = f"<{classes_ranges[0]}"
        elif n == n_classes-1:
            LabelsMap[n] = f"+{classes_ranges[-1]}"
        else:
            LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"

    df_parallel['OverallQual'] = df_parallel['OverallQual'].replace(LabelsMap)
    fig = px.parallel_categories(
        df_parallel, color="SalePrice", width=750, height=500)
    # we use st.plotly_chart() to render, in notebook is fig.show()
    st.plotly_chart(fig)