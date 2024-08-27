# Bank-Risk-Controller-Systems
# Singapore Resale Flat Prices Prediction

## Overview
The Singapore Resale Flat Prices Prediction system is a powerful predictive model developed to estimate the resale prices of Housing and Development Board (HDB) flats in Singapore. This tool is designed to assist both potential buyers and sellers in making informed decisions by providing accurate price predictions based on various flat features. The system utilizes machine learning techniques and is deployed as a user-friendly web application using Streamlit.

## Project Structure

### Main Libraries Used:
- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **streamlit**: Web application framework.
- **plotly.express**: Interactive data visualizations.
- **matplotlib & seaborn**: Data visualization.
- **scikit-learn**: Machine learning algorithms.

### File Structure:
- **app.py**: The primary Python file containing the Streamlit app code.
- **resale_flat_data.csv**: The dataset used for model training and analysis.
- **resale_price_model.pkl**: The pre-trained regression model for flat price prediction.


### Home
- **Overview**: Provides an introduction to the Singapore Resale Flat Prices Prediction system, including the project's purpose and objectives.
- **Information**: Displays details about the domain, technologies used, and the overall structure of the project.

### EDA Visualization
- **Exploratory Data Analysis**: Offers insights into the dataset, including distribution analysis, feature relationships, and handling of missing data.
- **Visualizations**: Includes histograms, correlation matrices, and scatter plots to better understand the data.

### Model Performance
- **Metrics**: Showcases the model's performance using regression metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score.
- **Model Selection**: Highlights the selected regression model (e.g., Random Forest Regressor) with a performance summary.

### Price Prediction
- **User Input**: A form-based interface allowing users to input flat details such as town, flat type, floor area, etc., to predict the resale price.
- **Prediction Output**: Displays the predicted resale price based on the user-provided data.

## Dataset
The dataset includes historical resale flat transactions in Singapore, with features such as flat type, town, storey range, floor area, and lease commence date. The target variable is the resale price.

## How to Use
1. **Navigate**: Use the sidebar menu to explore different sections of the app.
2. **Price Prediction**: Input flat details in the Price Prediction section to estimate the resale value.
3. **EDA Visualization**: Explore data characteristics and relationships through various visualizations.
4. **About Section**: Learn more about the project background, dataset, and methodologies used.

## Conclusion
The Singapore Resale Flat Prices Prediction system serves as an effective tool for estimating the resale value of flats in Singapore. The Streamlit application provides an intuitive interface for interacting with the predictive model, making it accessible to users who wish to gain insights into the resale flat market.

