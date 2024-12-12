## CoPulse Application

### Overview

A web application in the works that uses machine learning algorithm to predict heart disease. A simple web application which uses Machine Learning algorithm to predict the heart disease using person's characterics such as gender, blood pressure, cholestrol, heart rate, etc. 

### Project Structure

- Project consist `app.py` script which is used to run the application and is engine of this app. contians API that gets input from the user and computes a predicted value based on the model.
- `pulse-intelligence.py` code to build and train the AI model using kaggle dataset - processed_cleveland.csv
- main.py - primary business logic code to derive the prediction based on heart rate and other input information
- /templates - contains main.html and result.html - first used to gather input and second used to display the results. Both are connected via the Flask python calls.
- /static folder contains file `style.css` which adds some styling and enhance the look of the application. 

### Installation

The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/).

```
pip install -r requirements.txt 
```

### Run 

*To Run the Application*

```
python main.py
```

### Credit Source
 Credit: This script is based on and inspired by the work in the repository  
 https://github.com/asthasharma98/Heart-Disease-Prediction-Deployment.git  
 The html input form is also derived from the initial version of this project.








  
  
  


