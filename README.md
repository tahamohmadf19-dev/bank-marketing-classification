# Bank Marketing Campaign Classification

**Author:** Mohammad Taha — November 2025

## Project Overview

This project implements a comprehensive **machine learning classification system** to predict whether a bank client will subscribe to a term deposit based on direct marketing campaign data. Using **Apache Spark MLlib** and advanced data preprocessing techniques, the system achieves over **89% accuracy** in predicting customer responses.

The project demonstrates end-to-end big data processing, from data cleaning and feature engineering to model training, hyperparameter tuning, and evaluation using industry-standard metrics.

## Business Context

Direct marketing campaigns are costly and time-consuming. Banks need to identify which customers are most likely to respond positively to term deposit offers. This predictive model enables:

- **Targeted marketing**: Focus resources on high-probability customers
- **Cost reduction**: Minimize wasted effort on unlikely prospects
- **Campaign optimization**: Improve conversion rates through data-driven insights
- **Resource allocation**: Better planning of call center operations

## Dataset

**Source**: Bank Marketing Dataset (UCI Machine Learning Repository)  
**Size**: 45,211 records  
**Features**: 17 attributes including demographic, economic, and campaign-related variables

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Client's age |
| `job` | Categorical | Type of job (management, technician, entrepreneur, etc.) |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Binary | Has credit in default? |
| `balance` | Numeric | Average yearly balance (euros) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `duration` | Numeric | Last contact duration (seconds) |
| `campaign` | Numeric | Number of contacts during this campaign |
| `pdays` | Numeric | Days since last contact from previous campaign |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous marketing campaign |
| **`y`** | **Binary (Target)** | **Has the client subscribed to a term deposit?** |

## Technologies and Tools

- **Apache Spark 3.x**: Distributed data processing
- **PySpark MLlib**: Machine learning pipelines
- **Python 3.x**: Core programming language
- **Matplotlib**: Data visualization
- **Google Colab**: Development environment

## Methodology

### 1. Data Preprocessing

- **Missing value handling**: Replaced "unknown" values with "Not specified"
- **Label encoding**: Converted target variable (`y`) to binary (0/1)
- **Data splitting**: 60% training, 20% validation, 20% test (stratified)
- **Class distribution analysis**: Addressed imbalanced dataset (88% negative, 12% positive)

### 2. Feature Engineering

- **String Indexing**: Converted categorical variables to numeric indices
- **One-Hot Encoding**: Created binary vectors for categorical features
- **Vector Assembly**: Combined all features into a single feature vector
- **Imputation**: Handled any remaining missing values

### 3. Model Training and Evaluation

Three classification algorithms were trained and compared:

#### Decision Tree Classifier
- Simple interpretable model
- Baseline for comparison

#### Linear SVC (Support Vector Classifier)
- Linear decision boundary
- Efficient for high-dimensional data

#### Random Forest Classifier (Best Model)
- Ensemble of decision trees
- Handles non-linear relationships
- Robust to overfitting

### 4. Hyperparameter Tuning

Performed grid search on Random Forest with the following parameters:

| Parameter | Values Tested |
|-----------|---------------|
| `numTrees` | 50, 100, 200 |
| `maxDepth` | 5, 10, 15 |

**Best configuration**: 100 trees, max depth 15

## Results

### Model Performance Comparison

| Model | AUC (Validation) | Accuracy | F1 Score |
|-------|------------------|----------|----------|
| Decision Tree | 0.8756 | 0.8843 | 0.8384 |
| Linear SVC | 0.8917 | 0.8943 | 0.8516 |
| **Random Forest (Best)** | **0.9308** | **0.9083** | **0.8942** |

### Final Test Set Performance

The optimized Random Forest model achieved:

- **AUC**: 0.9221
- **Accuracy**: 89.88%
- **F1 Score**: 0.8831

### Key Insights

- **High AUC (0.92)**: Excellent ability to distinguish between positive and negative cases
- **Balanced F1 Score**: Good precision-recall trade-off despite class imbalance
- **Generalization**: Consistent performance across train, validation, and test sets
- **Feature importance**: Duration of last contact and previous campaign outcomes are strong predictors

## Project Structure

```
bank-marketing-classification/
├── bank_marketing_classification.ipynb  # Main Jupyter notebook
├── README.md                            # This file
├── LICENSE                              # MIT License
├── requirements.txt                     # Python dependencies
└── .gitignore                           # Git ignore rules
```

## How to Run

### Prerequisites

- Python 3.8+
- Java 8+ (required for Spark)
- Minimum 4GB RAM

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tahamohmadf19-dev/bank-marketing-classification.git
   cd bank-marketing-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Place `bank-full.csv` in the project directory
   - Dataset available from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

4. Run the notebook:
   ```bash
   jupyter notebook bank_marketing_classification.ipynb
   ```

### Running on Google Colab

1. Upload the notebook to Google Colab
2. Upload `bank-full.csv` to Colab's `/content/` directory
3. Run all cells sequentially

## Future Enhancements

- **Advanced algorithms**: Test XGBoost, LightGBM, and Neural Networks
- **Feature selection**: Use recursive feature elimination to reduce dimensionality
- **Cost-sensitive learning**: Incorporate business costs of false positives/negatives
- **Real-time prediction API**: Deploy model as a REST API using Flask/FastAPI
- **A/B testing framework**: Integrate with campaign management systems
- **Explainability**: Add SHAP values for individual prediction interpretation

## Educational Value

This project demonstrates:

- Distributed computing with Apache Spark
- Handling imbalanced classification problems
- End-to-end ML pipeline construction
- Model selection and hyperparameter tuning
- Business-oriented performance metrics
- Big data best practices

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.
- UCI Machine Learning Repository: Bank Marketing Dataset
- Apache Spark MLlib Documentation

## Contact

**Mohammad Taha**  
📧 Email: tahamohmadf19@gmail.com  
🔗 LinkedIn: [Mohmad Taha Alhmad](https://www.linkedin.com/in/mohmad-taha-alhmad)  
🐙 GitHub: [@tahamohmadf19-dev](https://github.com/tahamohmadf19-dev)

---

*This project was developed as part of a big data analytics portfolio to demonstrate practical machine learning skills in a business context.*
