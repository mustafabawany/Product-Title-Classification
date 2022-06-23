
# Product Title Classification

## Problem Statement

The goal of the project is to classify product title in an appropriate category which can assist
e-commerece sellers in listing products categories.

## Description

Product title classication is merely an instance of text classication problems, which are well-studied
in literature. However, product titles possess some properties very different from general
documents. A title is usually a very short description, and an incomplete sentence. A product title
classier may need to be designed differently from a text classier. 
<br>
We will do exploratory data analysis on the dataset to remove noisy data, then we will perform
feature selection and extraction to identify suitable algorithms for multi-class classification. The
observations and results will be put into production to make it available for the end-users.

<br>

<!-- ![Finished Website](https://github.com/mustafabawany/Boolean-Retrieval-Model/blob/main/Project_Demo.gif) -->

## How to execute
NOTE: You must have python pre-installed in your system
1. Clone this project on your local repository
```
git clone <repository link>
```
2. Install virtual environment in your system
```
pip3 install virtualenv
```
3. Create virtual environment
```
virtualenv env
```
4. Activate your virtual environment
```
source env/bin/activate
```
5. Install the required packages
```
pip3 install flask, numpy, nltk, matplotlib, pandas, scikit-learn
```
6. Execute the following command to run the program
```
python3 app.py
```

## Tools Used

- Python
- Flask
- NLTK
- HTML
- CSS
- Bootstrap

## Models Implemented
- Support Vector Machine (SVM)
- Random Forest Classifier 

## Software Used
- Jupyter Notebook
- Visual Studio Code
