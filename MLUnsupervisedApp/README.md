# Global Development Cluster Explorer

## Project Overview

The Global Development Cluster Explorer is an interactive Streamlit app that uses unsupervised machine learning to explore patterns in country-level development data.

Instead of predicting a target variable, this app groups countries based on similarities across socioeconomic, health, demographic, and economic indicators. Users can explore how countries cluster together using K-means clustering, hierarchical clustering, and Principal Component Analysis (PCA).

This project was created as part of my Data Science Portfolio final project.

## Live App

[Add your Streamlit app link here]

## Repository

[Add your GitHub repository link here]

## App Features

The app allows users to:

- Use a built-in country development dataset
- Upload their own CSV dataset
- Select numeric variables for analysis
- View non-numeric columns that are excluded from modeling
- Standardize numeric variables before analysis
- Run K-means clustering
- Run hierarchical clustering
- Explore PCA results
- View PCA scatterplots
- View cluster summary tables
- View cluster counts
- View a dendrogram for hierarchical clustering
- Interpret PCA loadings and explained variance

## Dataset

The built-in dataset is `Country-data.csv`, which contains country-level development indicators.

The dataset includes the following columns:

| Variable | Description |
|---|---|
| `country` | Country name |
| `child_mort` | Child mortality rate |
| `exports` | Exports as a percentage of GDP |
| `health` | Health spending as a percentage of GDP |
| `imports` | Imports as a percentage of GDP |
| `income` | Net income per person |
| `inflation` | Annual inflation rate |
| `life_expec` | Average life expectancy |
| `total_fer` | Total fertility rate |
| `gdpp` | GDP per capita |

The `country` column is used as an optional label, but it is not used as a model input. The unsupervised models use only numeric development indicators.

## Data Requirements for Uploaded Files

Users can upload their own CSV file.

Uploaded datasets should contain at least two numeric columns. The app uses only numeric variables for clustering and PCA.

Non-numeric columns are not used as model inputs. However, one non-numeric column can be selected as an optional label column. For example, a column containing country names, school names, company names, or observation IDs could be used to label points in the visualization.

If users want categorical variables included in the analysis, they should encode those variables before uploading the dataset.

For example, a categorical variable such as:

```text
Region: Europe, Africa, Asia
