# Can We Trust an AI Prediction?
### Comparing Random and Chronological Evaluation for Bike Rental Demand

## 📌 About the Project

Can we really trust an AI model just because it achieves a high accuracy score?

This project investigates how different data-splitting methods affect the performance evaluation of a Machine Learning model.

The study focuses on predicting hourly bike rental demand using the UCI Bike Sharing Dataset. It compares two evaluation methods:

- Random Train/Test Split
- Chronological Train/Test Split

The main objective is to understand whether a model's performance score accurately represents its ability to predict future data.

## 🎯 Research Question

How do performance estimates differ between a random train/test split and a chronological future-period test when predicting hourly bike rental demand?

## 📊 Dataset

This project uses the Hourly Bike Sharing Dataset from the UCI Machine Learning Repository.

- Dataset: Bike Sharing Dataset
- File: `hour.csv`
- Total Records: 17,379
- Total Columns: 17
- Prediction Target: `cnt` (Total hourly bike rentals)

Dataset source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Matplotlib
- Machine Learning

## ⚙️ Methodology

A Random Forest Regressor with 100 trees was used for the experiments.

**Experiment 1 – Random Split**

- 80% training data
- 20% testing data
- Random state: 42

**Experiment 2 – Chronological Split**

- First 80% of observations used for training.
- Final 20% used for testing.
- Data ordered by date and hour.

**Experiment 3 – Time Period Comparison**

Two chronological experiments were performed:

- First 60% for training and next 20% for testing.
- First 80% for training and final 20% for testing.

## 📈 Results

The model produced the following results:

| Evaluation Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Split | 24.903 | 42.069 | 0.944 |
| Chronological Split | 44.991 | 69.557 | 0.900 |


## 📈 Results and Visualizations

### 1. MAE Comparison

![MAE Comparison](mae_comparison.png)

### 2. Final MAE Comparison

![Final MAE Comparison](final_mae_comparison.png)

### 3. RMSE Comparison

![RMSE Comparison](rmse_comparison.png)

### 4. R² Comparison

![R2 Comparison](r2_comparison.png)

### 5. Hourly Prediction Errors

![Hourly Error Chart](hourly_error_chart.png)

### 6. Time Period Comparison

![Time Period MAE](time_period_mae.png)

### 7. Walk Forward Evaluation

![Walk Forward MAE](walk_forward_mae.png)


### Key Findings

- The random split achieved lower prediction errors.
- The chronological split produced higher prediction errors.
- Chronological MAE was approximately 80.7% higher than random-split MAE.
- The results suggest that random splitting can give an overly optimistic performance estimate when predicting future demand.

## 💡 Conclusion

A high Machine Learning performance score does not always mean that a model will perform equally well in real-world situations.

This research shows that evaluation methods matter. When the goal is to predict future demand, chronological evaluation provides a more realistic test of model performance.

However, further experiments using multiple future periods are needed to confirm the findings more strongly.

**The main lesson: We should not only ask how accurate an AI model is. We should also ask how it was tested.**

## ⚠️ Limitations

- Only one dataset was used.
- Only one model configuration was evaluated.
- Random and chronological test sets contain different observations.
- More repeated chronological experiments are needed.
- The study is an educational research comparison, not a complete production forecasting assessment.

## 📚 References

1. UCI Machine Learning Repository – Bike Sharing Dataset  
   https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

2. Scikit-learn – Model Evaluation  
   https://scikit-learn.org/stable/modules/model_evaluation.html

3. Scikit-learn – train_test_split  
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

## 👨‍💻 Author

Computer Science Undergraduate

GitHub: https://github.com/Sithum0000

---

*This project was developed for educational and research purposes to understand Machine Learning model evaluation and prediction reliability.*
