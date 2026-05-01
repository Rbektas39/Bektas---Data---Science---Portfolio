import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


st.set_page_config(
    page_title="Global Development Cluster Explorer",
    layout="wide"
)

st.title("Global Development Cluster Explorer")

st.write(
    "This app uses unsupervised machine learning to explore patterns in country-level development data. "
    "Instead of predicting a target variable, the app groups countries based on similarities across socioeconomic, "
    "health, demographic, and economic indicators."
)

with st.expander("How this app works"):
    st.write("""
    - Choose the built-in country development dataset or upload your own CSV file.
    - The app uses numeric columns for clustering and PCA.
    - Non-numeric columns are not used as model inputs, but one text column can be used as an identifier label.
    - Missing numeric values are filled using column means.
    - Numeric variables are standardized before clustering and PCA.
    - You can explore K-means clustering, hierarchical clustering, and principal component analysis.
    """)


@st.cache_data
def load_country_data():
    return pd.read_csv("Country-data.csv")


def prepare_data(df):
    """
    Prepares a dataset for unsupervised learning.

    The function separates numeric columns from non-numeric columns.
    Numeric columns are used for clustering and PCA.
    Non-numeric columns are not used in modeling, but they can be used as labels.
    Missing numeric values are filled using column means.
    """
    numeric_df = df.select_dtypes(include=np.number).copy()
    non_numeric_cols = [col for col in df.columns if col not in numeric_df.columns]

    if numeric_df.isnull().sum().sum() > 0:
        numeric_df = numeric_df.fillna(numeric_df.mean())

    return numeric_df, non_numeric_cols


def scale_data(numeric_df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    return scaled_df


def run_pca(scaled_df, n_components=2):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_df)

    component_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(
        components,
        columns=component_cols,
        index=scaled_df.index
    )

    return pca, pca_df


def plot_pca_clusters(pca_df, labels, title, label_series=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=labels,
        alpha=0.75
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best"
    )
    ax.add_artist(legend)

    if label_series is not None:
        for i in range(len(pca_df)):
            ax.annotate(
                str(label_series.iloc[i]),
                (pca_df["PC1"].iloc[i], pca_df["PC2"].iloc[i]),
                fontsize=7,
                alpha=0.65
            )

    st.pyplot(fig)


def plot_explained_variance(pca):
    explained_variance = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [f"PC{i+1}" for i in range(len(explained_variance))],
        explained_variance
    )
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Explained Variance by Principal Component")

    st.pyplot(fig)


def plot_dendrogram(scaled_df, linkage_method):
    linked = linkage(scaled_df, method=linkage_method)

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linked,
        ax=ax,
        truncate_mode="level",
        p=5
    )
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Distance")

    st.pyplot(fig)


def cluster_summary(original_numeric_df, labels):
    summary_df = original_numeric_df.copy()
    summary_df["Cluster"] = labels

    cluster_profiles = summary_df.groupby("Cluster").mean().round(2)
    cluster_counts = summary_df["Cluster"].value_counts().sort_index()

    return cluster_profiles, cluster_counts


def pca_loadings_table(pca, feature_names):
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_names
    )

    return loadings.round(3)


st.sidebar.header("App Controls")

data_option = st.sidebar.radio(
    "Choose a dataset source:",
    ["Use built-in country development dataset", "Upload my own CSV"]
)

if data_option == "Use built-in country development dataset":
    df = load_country_data()

else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.warning(
            "This app uses only numeric columns for clustering and PCA. "
            "Non-numeric columns are not used as model inputs. "
            "If you want categorical variables included, please encode them before uploading your dataset."
        )

    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()


st.subheader("Dataset Preview")
st.dataframe(df.head())

numeric_df, non_numeric_cols = prepare_data(df)

if len(non_numeric_cols) > 0:
    st.info(
        "Non-numeric columns not used as model inputs: "
        + ", ".join(non_numeric_cols)
    )

if len(numeric_df.columns) < 2:
    st.error("The dataset must contain at least two numeric columns for clustering and PCA.")
    st.stop()


st.sidebar.subheader("Feature Selection")

selected_features = st.sidebar.multiselect(
    "Select numeric variables to include:",
    options=numeric_df.columns.tolist(),
    default=numeric_df.columns.tolist()
)

if len(selected_features) < 2:
    st.error("Please select at least two numeric variables.")
    st.stop()

model_numeric_df = numeric_df[selected_features]

label_column = None

if len(non_numeric_cols) > 0:
    label_column = st.sidebar.selectbox(
        "Optional label column:",
        options=["None"] + non_numeric_cols,
        index=1 if "country" in non_numeric_cols else 0
    )

label_series = None
if label_column != "None" and label_column is not None:
    label_series = df[label_column]


st.subheader("Preprocessing Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Numeric Features Used", len(selected_features))
col3.metric("Missing Numeric Values", int(model_numeric_df.isnull().sum().sum()))

st.write(
    "Before running the models, selected numeric features are standardized. "
    "Standardization gives each variable a mean of 0 and standard deviation of 1, "
    "which prevents variables with larger scales, such as GDP per capita or income, from dominating the clustering results."
)

scaled_df = scale_data(model_numeric_df)

tabs = st.tabs([
    "K-means Clustering",
    "Hierarchical Clustering",
    "PCA"
])


with tabs[0]:
    st.header("K-means Clustering")

    st.write(
        "K-means clustering groups observations into a chosen number of clusters. "
        "The model tries to place observations into groups so that countries within the same cluster are similar to each other."
    )

    k = st.slider(
        "Number of clusters for K-means",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    st.caption(
        "The number of clusters controls how many groups the model creates. "
        "A smaller number gives broader groups, while a larger number creates more detailed groups."
    )

    kmeans_random_state = st.number_input(
        "K-means random state",
        min_value=0,
        max_value=1000,
        value=42,
        step=1
    )

    st.caption(
        "The random state makes the clustering result reproducible."
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=kmeans_random_state,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(scaled_df)

    pca_2, pca_2_df = run_pca(scaled_df, n_components=2)

    st.subheader("K-means Cluster Visualization")

    show_labels_kmeans = st.checkbox(
        "Show observation labels on K-means plot",
        value=False
    )

    plot_pca_clusters(
        pca_2_df,
        kmeans_labels,
        "K-means Clusters Visualized with PCA",
        label_series if show_labels_kmeans else None
    )

    st.subheader("Cluster Counts")

    kmeans_profiles, kmeans_counts = cluster_summary(
        model_numeric_df,
        kmeans_labels
    )

    st.dataframe(
        kmeans_counts.rename("Number of Observations").to_frame()
    )

    st.subheader("Cluster Profile Table")

    st.write(
        "This table shows the average value of each selected variable within each cluster. "
        "Use it to interpret what makes each group different."
    )

    st.dataframe(kmeans_profiles)

    if label_series is not None:
        st.subheader("Observations by Cluster")

        clustered_df = pd.DataFrame({
            label_column: label_series,
            "Cluster": kmeans_labels
        }).sort_values("Cluster")

        st.dataframe(clustered_df)


with tabs[1]:
    st.header("Hierarchical Clustering")

    st.write(
        "Hierarchical clustering builds groups based on distances between observations. "
        "Unlike K-means, it creates a tree-like structure that shows how observations merge into larger groups."
    )

    h_clusters = st.slider(
        "Number of clusters for hierarchical clustering",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    st.caption(
        "This controls how many final groups are cut from the hierarchical clustering tree."
    )

    linkage_method = st.selectbox(
        "Linkage method",
        options=["ward", "complete", "average", "single"],
        index=0
    )

    st.caption(
        "The linkage method controls how distances between groups are calculated. "
        "Ward often creates compact clusters, while complete, average, and single use different distance rules."
    )

    hierarchical = AgglomerativeClustering(
        n_clusters=h_clusters,
        linkage=linkage_method
    )

    hierarchical_labels = hierarchical.fit_predict(scaled_df)

    pca_2_h, pca_2_h_df = run_pca(scaled_df, n_components=2)

    st.subheader("Dendrogram")

    st.write(
        "The dendrogram shows how observations are gradually merged into clusters. "
        "Large vertical jumps can suggest natural separations in the data."
    )

    plot_dendrogram(scaled_df, linkage_method)

    st.subheader("Hierarchical Cluster Visualization")

    show_labels_h = st.checkbox(
        "Show observation labels on hierarchical plot",
        value=False
    )

    plot_pca_clusters(
        pca_2_h_df,
        hierarchical_labels,
        "Hierarchical Clusters Visualized with PCA",
        label_series if show_labels_h else None
    )

    st.subheader("Cluster Counts")

    h_profiles, h_counts = cluster_summary(
        model_numeric_df,
        hierarchical_labels
    )

    st.dataframe(
        h_counts.rename("Number of Observations").to_frame()
    )

    st.subheader("Cluster Profile Table")

    st.write(
        "This table shows the average value of each selected variable within each hierarchical cluster."
    )

    st.dataframe(h_profiles)

    if label_series is not None:
        st.subheader("Observations by Cluster")

        h_clustered_df = pd.DataFrame({
            label_column: label_series,
            "Cluster": hierarchical_labels
        }).sort_values("Cluster")

        st.dataframe(h_clustered_df)


with tabs[2]:
    st.header("Principal Component Analysis")

    st.write(
        "Principal Component Analysis, or PCA, reduces many numeric variables into a smaller number of components. "
        "These components summarize the main patterns of variation in the data."
    )

    max_components = min(len(selected_features), 10)

    n_components = st.slider(
        "Number of PCA components",
        min_value=2,
        max_value=max_components,
        value=2,
        step=1
    )

    st.caption(
        "More components preserve more information from the original variables. "
        "Two components are useful for visualization, while additional components can explain more total variance."
    )

    pca_model, pca_df = run_pca(scaled_df, n_components=n_components)

    st.subheader("Explained Variance")

    st.write(
        "Explained variance shows how much information each principal component captures from the original variables."
    )

    plot_explained_variance(pca_model)

    explained_variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(n_components)],
        "Explained Variance Ratio": pca_model.explained_variance_ratio_,
        "Cumulative Explained Variance": np.cumsum(pca_model.explained_variance_ratio_)
    })

    st.dataframe(explained_variance_df.round(3))

    st.subheader("PCA Scatterplot")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        alpha=0.75
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Projection")

    show_labels_pca = st.checkbox(
        "Show observation labels on PCA plot",
        value=False
    )

    if show_labels_pca and label_series is not None:
        for i in range(len(pca_df)):
            ax.annotate(
                str(label_series.iloc[i]),
                (pca_df["PC1"].iloc[i], pca_df["PC2"].iloc[i]),
                fontsize=7,
                alpha=0.65
            )

    st.pyplot(fig)

    st.subheader("PCA Loadings")

    st.write(
        "Loadings show how strongly each original variable contributes to each principal component. "
        "Large positive or negative values suggest that a variable is important for that component."
    )

    loadings_df = pca_loadings_table(
        pca_model,
        selected_features
    )

    st.dataframe(loadings_df)

    st.subheader("Interpretation Note")

    st.write(
        "The clusters and PCA plots should not be interpreted as fixed rankings of countries. "
        "They are exploratory similarity patterns based on the selected variables, scaling choices, and model settings."
    )