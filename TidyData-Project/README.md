# Tidy Data Project: 2008 Olympic Medalists

## Project Overview

This project demonstrates how messy real-world data can be transformed into tidy format using Python and pandas. The dataset contains medal winners from the 2008 Olympic Games. In the original dataset, sport and gender information were embedded within column names, making analysis difficult.

Using tidy data principles, the dataset was reshaped into a long format where each variable has its own column and each observation represents a single medal result. This allowed for easier aggregation, visualization, and interpretation.

## Dataset Description

The dataset contains information on medalists from the 2008 Olympic Games. Each row originally represented an athlete, and medal results were spread across multiple sport-gender columns.

Pre-processing steps included:
- Reshaping the dataset using pandas `melt()`
- Removing missing observations
- Splitting combined variables using `str.split()`
- Cleaning text values using `str.replace()` and `str.strip()`

## Visualizations and Analysis

After cleaning, the tidy dataset was used to:
- Create pivot tables summarizing medal counts by sport and gender
- Visualize the top sports by number of medalists
- Compare medal distributions between male and female athletes

## How to Run This Project

1. Clone this repository
2. Install required dependencies:

- Pandas Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf  
- Wickham, H. (2014). Tidy Data. Journal of Statistical Software.

