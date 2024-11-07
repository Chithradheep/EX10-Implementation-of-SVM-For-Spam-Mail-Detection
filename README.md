# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Preprocess the email text by tokenizing, removing stop words, and converting to a numerical feature vector (e.g., TF-IDF).
2. Label emails as "spam" or "ham" (not spam) for supervised training.
3. Train the SVM model on the labeled dataset, finding a hyperplane that maximizes the margin between "spam" and "ham" classes.
4. Evaluate the model’s accuracy using a validation set and adjust parameters if needed.
5. Classify new emails as "spam" or "ham" based on the trained model’s prediction.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: Chithradheep R
RegisterNumber:  2305002003

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv("/content/spamEX10.csv",encoding='ISO-8859-1')
df.head()
vectorizer = CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)
model = svm.SVC()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("ACCUARACY:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]
new_message = "Congratulations!"
result = predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
*/
```

## Output:
![Screenshot 2024-11-07 100803](https://github.com/user-attachments/assets/d81761b8-8516-4b7d-8184-c527683b00a8)
![Screenshot 2024-11-07 100847](https://github.com/user-attachments/assets/fac4ac2e-7b08-4c7b-bac8-847cebb131a3)
![Screenshot 2024-11-07 100905](https://github.com/user-attachments/assets/ae371017-c447-4ca7-8976-84a1a85c172c)
![Screenshot 2024-11-07 100937](https://github.com/user-attachments/assets/cd6471bf-8409-4521-9ed7-f300a97338e2)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
