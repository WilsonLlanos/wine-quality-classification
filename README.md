# wine-quality-classification

🍷 Wine Classification with Machine Learning
This project aims to classify red wines based on their physicochemical properties using machine learning algorithms. The entire development was carried out on the Databricks platform, covering the full ML pipeline: from data cleaning to model deployment and tracking with MLflow.

🔍 Objective
Predict wine quality (classified into 3 categories: low, medium, or high) based on chemical attributes using supervised machine learning.

⚙️ Technologies and Libraries
Databricks (Apache Spark environment)

Python

Pandas / PySpark

Scikit-learn

Seaborn / Matplotlib / Sweetviz

MLflow (for tracking and model registration)

SMOTE (for data balancing)

Random Forest Classifier

🧪 Development Pipeline
Exploratory data analysis (visualizations and Sweetviz profiling)

Outlier and missing value treatment

Reclassification of the target variable (quality) into 3 categories

Data balancing using SMOTE

Feature scaling using StandardScaler

Training models: Logistic Regression and Random Forest

Hyperparameter tuning with RandomizedSearchCV

Model evaluation using metrics such as Accuracy and ROC AUC

Model registration and signature tracking using MLflow

Prediction testing with new samples

📊 Results
Final accuracy with Random Forest: 86.8%

ROC AUC score: 91.3%

Model registered and versioned with MLflow

Model able to generalize well for new input samples

🧠 Key Learnings
Although the first model tested was Logistic Regression, it did not perform well with the multi-class problem. Instead of overfitting this simpler model, I switched to Random Forest, which significantly improved the results. This decision emphasized the importance of experimenting with different models and validating them with proper metrics.

📁 Project Access
This project was entirely developed on Databricks. You can:

Access the notebook: [./notebooks/classificacao-vinhos.ipynb](https://github.com/WilsonLlanos/wine-quality-classification/blob/main/wine-quality-classification.ipynb)

Review the registered model and metrics through the MLflow dashboard

📎 Dataset
Red Wine Quality Dataset
Available at: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download

📬 Contact
If you have questions, feedback, or opportunities to collaborate, feel free to reach out via LinkedIn.
