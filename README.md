# College Chatbot Using ML Algorithm and NLP Toolkit

## Aim of a Chatbot
The aim of a chatbot is to provide automated assistance to users in a conversational manner. A chatbot can be designed to perform various tasks such as answering questions, providing recommendations, scheduling appointments, making reservations, and more. The primary goal of a chatbot is to improve the user experience by providing quick and accurate responses to user inquiries. Additionally, a chatbot can help businesses save time and resources by automating repetitive tasks and providing 24/7 customer support. Ultimately, the aim of a chatbot is to provide a seamless and efficient communication channel between users and businesses.


## Methodology

This is a Python code for creating a chatbot using Natural Language Processing (NLP) techniques. The code reads in a JSON file containing various intents for the chatbot and their corresponding responses. It preprocesses the data by tokenizing and removing stop words from the text examples, and then converts the text data into numerical form using the TfidfVectorizer from the scikit-learn library.

The code then trains and evaluates multiple machine learning models using GridSearchCV to find the best performing model based on accuracy. The selected model is then used to predict the intent of user inputs and generate responses based on the corresponding intent.

Finally, the best model, vectorizer, and intents data are saved using pickle and json, respectively, for future use. A chatbot is created by running a while loop that takes user inputs and returns chatbot responses until the user inputs the word "quit".

## Motivation
The motivation behind this project was to create a simple chatbot using my newly acquired knowledge of Natural Language Processing (NLP) and Python programming. As one of my first projects in this field, I wanted to put my skills to the test and see what I could create.

[I followed a guide referenced in the project](https://thecleverprogrammer.com/2023/03/27/end-to-end-chatbot-using-python/) to learn the steps involved in creating an end-to-end chatbot. This included collecting data, choosing programming languages and NLP tools, training the chatbot, and testing and refining it before making it available to users.

Although this chatbot may not have exceptional cognitive skills or be state-of-the-art, it was a great way for me to apply my skills and learn more about NLP and chatbot development. I hope this project inspires others to try their hand at creating their own chatbots and further explore the world of NLP.

## How to Use the Chatbot
You can run the [Chatbot.ipynb](https://github.com/roshancharlie/College-Chatbot-Using-ML-and-NLP/blob/main/Chatbot.ipynb) which also includes step by step instructions in [Jupyter Notebook](https://www.geeksforgeeks.org/how-to-install-jupyter-notebook-in-windows/).
### Or
You can run Chatbot Through Terminal
```
python Chatbot.py
```






