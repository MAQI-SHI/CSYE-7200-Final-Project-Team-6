# CSYE7200-Final-Project-Team6
![GitHub top language]

This project is designed as the final project of Northeastern University COE-CSYE7200 , taught by [Prof. Robin Hillyard](https://github.com/rchillyard).

## TeamMember

| Name        | NuID      |
| ----------- | --------- |
| [@Maqi Shi](https://github.com/MAQI-SHI) | 001057366 |
| [@Yue Liu](https://github.com/YL-Hurry) | 001353606 |

## Abstract

Our goal is to predict if someone will get stroke based on their health condition.The dataset is from [Kaggle](https://www.kaggle.com/lirilkumaramal/heart-stroke). We preprocessed the data and train 3 different models(Decision Tree, Radom Forest, Logistic Regression) with 7 features(smoking_status, age, etc). Our program can automatically select the best model base on accuracy. The users can input the information of their health condition and then get the result of the prediction.

## Getting Started

First , clone or download the repository to local.

Open the **final** file with ***IDEA***. Run ```Main.scala``` in ```/src/main/scala/Main.scala```.It will take some minutes to train all machine learning models when you first run ```Main.scala```. Follow the instruction in console to input required features and system will return predictions.


## Running the tests

Open the **final** files with ***IDEA***.

Run the tests ```sbt test``` in each terminals

## Built With

* [Scala](https://www.scala-lang.org/) - The program language to implement the program.
* [IntelliJ IDEA](https://www.jetbrains.com/idea/) - The IDE to development the system.
* [Spark](https://databricks.com/spark/about) - The framework to develop the Machine Learning process
