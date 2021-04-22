# CSYE7200-Final-Project-Team-6
![GitHub top language](https://img.shields.io/github/languages/top/MAQI-SHI/CSYE7200-Final-Project-Team-6.svg)

This project is the final project of Northeastern University CSYE7200 - Big Data System Engineering with Scala. The professor is [Prof. Robin Hillyard](https://github.com/rchillyard).

## TeamMember

[@Maqi Shi](https://github.com/MAQI-SHI) 001057366 [@shi.maq@northeastern.edu](shi.maq@norteastern.edu)
[@Yue Liu](https://github.com/YL-Hurry) 001353606 [@liu.yue7@northeastern.edu](liu.yue7@northestern.edu)

## Abstract

The goal of our project is to predict if someone will get stroke based on their health condition. The dataset is from [Kaggle](https://www.kaggle.com/lirilkumaramal/heart-stroke). We first preprocessed the data and than train 3 different models(Decision Tree, Radom Forest, Logistic Regression) with 7 features(smoking_status, age, etc). Our program will select the best model by comparing accuracy of these models. The users can input the information of their health condition and then get the prediction.

## Methodology



## How to Run

Clone or download the repository to local.

Open the **final** file with ***IDEA***. Run ```Main.scala``` in ```/src/main/scala/Main.scala```.It will take some time to train all machine learning models when you first run this project. Follow the instruction in console and input required features and system will return predictions.


## How to Test

Open the **final** files with ***IDEA***.

Run the ```sbt test``` in terminal to start test.

## Built With

* [Scala](https://www.scala-lang.org/) - The program language to implement the program.
* [IntelliJ IDEA](https://www.jetbrains.com/idea/) - The IDE to development the system.
* [Spark](https://databricks.com/spark/about) - The framework to develop the Machine Learning process
