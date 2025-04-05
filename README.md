
# 📊 Business Intelligence: Data Exploration

This project explores accident-related datasets using Python notebooks and performs data wrangling, summarization, and time series analysis. It’s designed for business analysts or data scientists to extract insights and prepare visual or statistical summaries from raw data.

## 📁 Project Structure

```
business-intelligence-data-exploration/
├── source/                              # Jupyter Notebooks and data
│   ├── accident2024 + $Damage.csv       # Raw dataset
│   ├── count_total_accident.ipynb       # Aggregation of accident data
│   ├── fill_province.ipynb              # Province-filling logic
│   ├── time_series_decomposition.ipynb  # Time series trend analysis
│   ├── exploratory_data_anysis.ipynb    # Exploratory Data Analysis
│   ├── feature_engineering.ipynb        # Feature creation for modeling
│   └── master_data/                     # Reference datasets
├── output/                              # Cleaned and summarized outputs
│   ├── accident2024_sum_accident.csv
│   ├── accident2024_Damage_provinces.csv
│   └── accident2024_sum_accident.xlsx
├── clean_data_for_cross_section/        # Data prepared for modeling
├── .gitignore
├── .venv/                               # Virtual environment (optional)
└── .idea/                               # IDE settings (for PyCharm, etc.)
```

## 📌 Key Notebooks

| Notebook                          | Purpose                                    |
|----------------------------------|--------------------------------------------|
| `count_total_accident.ipynb`     | Count and summarize total accidents        |
| `fill_province.ipynb`            | Fill missing or incomplete province data   |
| `time_series_decomposition.ipynb`| Analyze time trends in accident data       |
| `exploratory_data_anysis.ipynb`  | Perform EDA to understand data distribution, trends, and relationships |
| `feature_engineering.ipynb`      | Generate new features and prepare data for modeling |

## 📦 Outputs

All summarized data is saved in the `output/` folder in `.csv` and `.xlsx` formats for further use in reporting or dashboards.
