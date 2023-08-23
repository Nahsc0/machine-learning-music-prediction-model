# Machine Learning Music Prediction Model

This repository contains a machine learning prediction model that uses the `sklearn.tree` DecisionTreeClassifier algorithm. The model predicts the type of music a person will like based on their age and gender.

## Steps

1. **Import Data**: The first step is to import the data for training the model. The dataset should have three columns: `age`, `gender`, and `genre`. The `age` column represents the age of the person, where `1` indicates male and `0` indicates female.

2. **Clean Data**: Before training the model, it's important to clean the data. This includes handling missing values, removing duplicates, and ensuring data consistency. You may also need to perform data transformations or feature engineering if required.

3. **Split Data**: Split the dataset into training and testing sets. The training set will be used to train the model, while the testing set will be used to evaluate its performance. The recommended train-test split ratio is usually 70:30 or 80:20, but you can adjust it based on your specific needs.

4. **Create a Model**: Use the DecisionTreeClassifier algorithm from the `sklearn.tree` module to create a machine learning model. This algorithm is suitable for classification problems, such as predicting the music genre based on age and gender. 

5. **Train the Model**: Fit the model to the training dataset. This step involves feeding the model with the training data, allowing it to learn the patterns and relationships between the input features (age and gender) and the target variable (music genre). 

6. **Make Predictions**: Finally, use the trained model to make predictions on new, unseen data. Pass the age and gender of an individual to the model, and it will predict the most likely music genre that person will enjoy.

## Tree Diagram

Below is a tree diagram illustrating the steps involved in building and using the machine learning music prediction model:

```
                    Import Data
                      /    \
            Clean Data     Split Data
              /   \          /   \
     Create Model  Train Model  Make Predictions
```

You can visualize the relationship between the steps and how they flow together to build the final model.

Feel free to explore the code in this repository to delve into the implementation details of each step.

## Usage

To use the machine learning music prediction model, follow these steps:

1. Clone the repository:

```shell
   git clone https://github.com/Nahsc0/machine-learning-music-prediction-model.git

   cd machine-learning-music-prediction-model

```


2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

3. Execute the model training script:

```shell
python train_model.py
```

4. After training, you can use the model to make predictions by running the prediction script:

```shell
python predict.py
```

Remember to provide the necessary inputs (age and gender) when prompted by the script.

Please refer to the documentation within the scripts for further details and customization options.

## Contributing

If you would like to contribute to this project, feel free to submit a pull request with your enhancements or bug fixes.

