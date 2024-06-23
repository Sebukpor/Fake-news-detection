# Fake News Detection

This repository contains a fake news detection model built using various machine learning algorithms including Logistic Regression, Decision Tree Classifier, Gradient Boosting Classifier, and Random Forest Classifier. The model is trained to classify news articles as either "Fake News" or "Not A Fake News".

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [License](#license)

## Installation

To install the necessary dependencies for this project, you can use the provided package installation commands.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sebukpor/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Install Python packages:**

    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

## Usage

To run the fake news detection model, follow these steps:

1. **Prepare the datasets:**

    Ensure you have `Fake.csv` and `True.csv` files containing the fake and true news articles respectively. Place these files in the appropriate directory.

2. **Run the model script:**

    ```bash
    python fake_news_detection.py
    ```

    The script will load the datasets, preprocess the text data, train the models, and provide classification reports for each model.

## Project Structure

- `fake_news_detection.py`: Main script containing the model and data processing logic.
- `README.md`: This README file.
- `Fake.csv`: CSV file containing fake news articles.
- `True.csv`: CSV file containing true news articles.

## Dependencies

The project requires the following dependencies:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Model Training and Evaluation

The fake news detection model is trained using the following steps:

1. **Data Preprocessing:** Cleaning and preprocessing the text data.
2. **Data Splitting:** Splitting the data into training and testing sets.
3. **Vectorization:** Transforming the text data using TfidfVectorizer.
4. **Model Training:** Training various models including Logistic Regression, Decision Tree Classifier, Gradient Boosting Classifier, and Random Forest Classifier.
5. **Evaluation:** Evaluating the models using classification reports and accuracy scores.

To train and evaluate the models, the script will:

- Load the datasets.
- Preprocess the text data.
- Split the data into training and testing sets.
- Train the models using the training set.
- Evaluate the models using the testing set.
- Provide classification reports for each model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by creating issues or pull requests.
