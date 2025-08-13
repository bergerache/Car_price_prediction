# 🚗 Used Car Price Prediction

**Machine Learning Regression Model | Python + Scikit-learn**

*Accurate price estimation for used vehicles using advanced machine learning techniques*

---

## 🎯 Business Problem & Solution

**Problem:** Used car buyers and sellers struggle to determine fair market prices, leading to financial losses, extended negotiation times, and market inefficiencies in the automotive resale market.

**Solution:** Advanced machine learning model that accurately predicts used car prices based on vehicle characteristics, enabling data-driven pricing decisions for both buyers and sellers in the automotive marketplace.

## 👥 Target Audience

**Primary Users:**
- **Car Dealerships** - Inventory pricing and trade-in valuations
- **Individual Sellers** - Fair market price estimation for private sales
- **Car Buyers** - Price verification and negotiation support
- **Insurance Companies** - Vehicle valuation for claims and coverage

---

## 📊 **Model Performance & Results**

**📈 [View Complete Analysis →](https://github.com/bergerache/Car_price_prediction/blob/main/Car%20Price%20Prediction.ipynb)**

**Dataset Specifications:**
- **Original Dataset**: 354,369 used car records
- **Final Clean Dataset**: 196,583 records after preprocessing and outlier removal
- **Features**: 9 predictive variables including VehicleType, RegistrationYear, Gearbox, Power, Model, Mileage, FuelType, Brand, NotRepaired
- **Target Variable**: Car price in euros

### **Model Performance Comparison**

| Model | RMSE (Cross-Validation) | RMSE (Test Set) | Runtime (seconds) |
|-------|------------------------|-----------------|-------------------|
| **Gradient Boosting** ⭐ | **1,491.4** | **1,495.8** | 0.4 |
| Random Forest | 1,884.5 | 1,870.0 | 0.4 |
| Linear Regression | 1,920.1 | 1,921.9 | 0.2 |
| ElasticNet | 1,939.5 | 1,938.5 | 0.1 |
| AdaBoost | 2,125.8 | 2,139.0 | 0.4 |

**🏆 Best Performing Model: Gradient Boosting Regressor**
- **Optimal Performance**: 1,495.8 RMSE on test set
- **Hyperparameters**: n_estimators=200
- **Error Rate**: ~37% relative to mean price (€4,002.5)

---

## 🔍 **Key Business Insights Generated**

### **Data Quality & Preprocessing Intelligence**
- **Missing Value Strategy**: Strategic removal of 20% incomplete records still retained 245,814 samples for robust training
- **Outlier Impact**: Systematic removal of price outliers (>€16,501) and unrealistic power values (>262.5 HP) improved model accuracy
- **Feature Selection**: Eliminated irrelevant features (NumberOfPictures, PostalCode, RegistrationMonth) to focus on car-specific attributes

### **Price Prediction Accuracy**
- **Gradient Boosting Excellence**: Achieved best performance with 22% improvement over baseline Linear Regression
- **Ensemble Methods Advantage**: Tree-based models (Random Forest, Gradient Boosting) significantly outperformed linear approaches
- **Real-World Application**: €1,495 average prediction error enables practical pricing decisions for automotive market

### **Performance vs Speed Trade-offs**
- **Optimal Balance**: Gradient Boosting provides best accuracy with reasonable 0.4-second runtime
- **Speed Champions**: Linear Regression and ElasticNet offer fastest predictions (0.1-0.2 seconds) with acceptable accuracy
- **Business Decision**: Gradient Boosting recommended for batch pricing; Linear Regression for real-time applications

---

## 🛠️ **Technical Implementation**

### **Data Processing Pipeline**
- **Initial Dataset**: 354,369 used car records with 13 features
- **Data Cleaning**: Systematic removal of missing values and irrelevant date columns
- **Outlier Treatment**: Statistical outlier removal using IQR method (1.5 * IQR beyond Q1/Q3)
- **Final Dataset**: 196,583 clean records with 9 predictive features

### **Feature Engineering & Preprocessing**
- **Numerical Features**: RegistrationYear, Power, Mileage (StandardScaler normalization)
- **Categorical Features**: VehicleType, Gearbox, Model, FuelType, Brand, NotRepaired (OneHotEncoder)
- **Pipeline Architecture**: Scikit-learn preprocessing pipelines for consistent data transformation
- **Train-Test Split**: 70/30 split with random_state=42 for reproducible results

### **Model Development Process**

#### **1. Baseline Model**
- **Linear Regression**: Established baseline performance (RMSE: 1,921.9)
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Runtime Efficiency**: Fastest prediction time (0.2 seconds)

#### **2. Advanced Algorithms**
- **ElasticNet**: Regularized linear model with hyperparameter tuning
- **Random Forest**: Ensemble method with max_depth and n_estimators optimization
- **AdaBoost**: Adaptive boosting with learning_rate tuning
- **Gradient Boosting**: Gradient-based ensemble with n_estimators optimization

#### **3. Hyperparameter Optimization**
- **HalvingGridSearchCV**: Efficient hyperparameter search with early elimination
- **Cross-Validation**: 3-fold CV for parameter selection
- **Scoring Metric**: Negative Mean Squared Error for optimization

---

## 📈 **Algorithm Performance Analysis**

### **Gradient Boosting Regressor (Best Performer)**
- **RMSE**: 1,495.8 (best accuracy)
- **Hyperparameters**: n_estimators=200
- **Strengths**: Excellent prediction accuracy, handles non-linear relationships
- **Use Case**: Primary model for accurate price estimation

### **Random Forest Regressor (Second Best)**
- **RMSE**: 1,870.0 (good accuracy)
- **Hyperparameters**: max_depth=5, n_estimators=100
- **Strengths**: Robust to overfitting, good interpretability
- **Use Case**: Alternative model for feature importance analysis

### **Linear Regression (Baseline)**
- **RMSE**: 1,921.9 (acceptable accuracy)
- **Strengths**: Fastest runtime, high interpretability
- **Use Case**: Real-time applications requiring instant predictions

---

## 🚀 **Getting Started**

### **Prerequisites**
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
```

### **Installation & Setup**
1. **Clone the repository**
   ```bash
   git clone https://github.com/bergerache/Car_price_prediction.git
   cd Car_price_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook "Car Price Prediction.ipynb"
   ```

### **Quick Prediction Example**
```python
# Load and preprocess data
df = pd.read_csv('car_data.csv')
# ... preprocessing steps as shown in notebook ...

# Train best model (Gradient Boosting)
model = GradientBoostingRegressor(n_estimators=200, random_state=42)
pipeline = make_pipeline(preprocessor, model)
pipeline.fit(X_train, y_train)

# Make prediction
sample_car = pd.DataFrame({
    'VehicleType': ['sedan'],
    'RegistrationYear': [2015],
    'Gearbox': ['auto'],
    'Power': [150],
    'Model': ['golf'],
    'Mileage': [80000],
    'FuelType': ['gasoline'],
    'Brand': ['volkswagen'],
    'NotRepaired': ['no']
})

predicted_price = pipeline.predict(sample_car)
print(f"Estimated Price: €{predicted_price[0]:,.0f}")
```

---

## 🎯 **Skills Demonstrated**

### **Machine Learning Engineering**
- ✅ **Algorithm Comparison**: Systematic evaluation of 5 different regression algorithms
- ✅ **Hyperparameter Optimization**: HalvingGridSearchCV for efficient parameter tuning
- ✅ **Pipeline Development**: End-to-end preprocessing and modeling pipelines
- ✅ **Model Validation**: Cross-validation and train-test evaluation protocols

### **Data Science & Preprocessing**
- ✅ **Data Cleaning**: Systematic missing value treatment and outlier removal
- ✅ **Feature Engineering**: Strategic feature selection and transformation
- ✅ **Statistical Analysis**: Outlier detection using IQR methodology
- ✅ **Data Pipeline**: Scalable preprocessing workflows with scikit-learn

### **Technical Skills**
- ✅ **Python Programming**: Advanced pandas, numpy, and scikit-learn implementation
- ✅ **Ensemble Methods**: Random Forest, Gradient Boosting, and AdaBoost implementation
- ✅ **Performance Optimization**: Runtime analysis and accuracy-speed trade-offs
- ✅ **Code Organization**: Clean, reproducible analysis with proper documentation

### **Business Intelligence**
- ✅ **Performance Metrics**: Comprehensive model evaluation and comparison
- ✅ **Business Impact**: Practical error rates and real-world applicability
- ✅ **Decision Framework**: Model selection based on business requirements
- ✅ **Scalability**: Solutions for both batch and real-time prediction scenarios

---

## 📊 **Business Impact & Applications**

### **For Car Dealerships**
- **Inventory Pricing**: Accurate valuation using Gradient Boosting model (€1,495 average error)
- **Trade-In Assessment**: Quick Linear Regression estimates for customer interactions
- **Competitive Analysis**: Data-driven pricing strategies based on vehicle characteristics
- **Profit Optimization**: Precise margin calculations with reliable price predictions

### **For Individual Sellers**
- **Fair Market Pricing**: Objective price estimation eliminating guesswork
- **Negotiation Support**: Data-backed pricing justification with model confidence
- **Market Positioning**: Understanding of price factors (year, mileage, brand impact)
- **Timing Decisions**: Depreciation insights for optimal selling strategies

### **For Car Buyers**
- **Price Verification**: Validation of asking prices against model predictions
- **Value Assessment**: Comprehensive evaluation considering all vehicle factors
- **Negotiation Strategy**: Data-driven approach to price discussions
- **Purchase Planning**: Accurate budgeting with reliable price estimates

---

## 💡 **Key Technical Achievements**

### **Advanced Data Processing**
- **Large-Scale Cleaning**: Successfully processed 354,369 records to 196,583 clean samples
- **Intelligent Feature Selection**: Removed irrelevant features (dates, postal codes) for improved performance
- **Robust Outlier Treatment**: Statistical approach to outlier removal preserving data integrity

### **Model Performance Optimization**
- **Algorithm Excellence**: Achieved 22% improvement over baseline with Gradient Boosting
- **Hyperparameter Mastery**: Systematic optimization using HalvingGridSearchCV
- **Validation Rigor**: Cross-validation ensuring model generalizability

### **Production-Ready Implementation**
- **Pipeline Architecture**: End-to-end preprocessing and prediction workflows
- **Runtime Efficiency**: Balanced accuracy and speed for different use cases
- **Scalable Design**: Framework suitable for both batch and real-time applications

---

## 📈 **Model Selection Rationale**

### **Why Gradient Boosting Won**
- **Superior Accuracy**: 22% better performance than baseline Linear Regression
- **Non-Linear Relationships**: Captures complex interactions between vehicle features
- **Robust Performance**: Consistent results across cross-validation folds
- **Business Value**: €426 lower average error compared to Random Forest

### **When to Use Alternatives**
- **Real-Time Applications**: Linear Regression for sub-second predictions
- **Interpretability Needs**: Random Forest for feature importance analysis
- **Resource Constraints**: ElasticNet for minimal computational requirements

---

*Part of a comprehensive Business Intelligence Portfolio demonstrating advanced machine learning applications for real-world automotive market solutions.*
