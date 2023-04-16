#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import random
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[2]:


# define the intents dictionary
intent = {
   "intents":[
 {
   "tag": "greeting",
   "patterns": [
      "Hi",
      "How are you?",
      "Is anyone there?",
      "Hello",
      "Good day",
      "What's up",
      "how are ya",
      "heyy",
      "whatsup",
      "??? ??? ??"
   ],
   "responses": [
      "Hello!",
      "Good to see you again!",
      "Hi there, how can I help?"
   ],
   "context_set": ""
},
 {
   "tag": "goodbye",
   "patterns": [
      "cya",
      "see you",
      "bye bye",
      "See you later",
      "Goodbye",
      "I am Leaving",
      "Bye",
      "Have a Good day",
      "talk to you later",
      "ttyl",
      "i got to go",
      "gtg"
   ],
   "responses": [
      "Sad to see you go :(",
      "Talk to you later",
      "Goodbye!",
      "Come back soon"
   ],
   "context_set": ""
},
 {
   "tag": "creator",
   "patterns": [
      "what is the name of your developers",
      "what is the name of your creators",
      "what is the name of the developers",
      "what is the name of the creators",
      "who created you",
      "your developers",
      "your creators",
      "who are your developers",
      "developers",
      "you are made by",
      "you are made by whom",
      "who created you",
      "who create you",
      "creators",
      "who made you",
      "who designed you"
   ],
   "responses": [
      "Roshan Sinha"
   ],
   "context_set": ""
},
 {
   "tag": "name",
   "patterns": [
      "name",
      "your name",
      "do you have a name",
      "what are you called",
      "what is your name",
      "what should I call you",
      "whats your name?",
      "what are you",
      "who are you",
      "who is this",
      "what am i chatting to",
      "who am i taking to",
      "what are you"
   ],
   "responses": [
      "You can call me Mind Reader.",
      "I'm Mind Reader",
      "I am a Chatbot.",
      "I am your helper"
   ],
   "context_set": ""
},
 {
   "tag": "hours",
   "patterns": [
      "timing of college",
      "what is college timing",
      "working days",
      "when are you guys open",
      "what are your hours",
      "hours of operation",
      "when is the college open",
      "college timing",
      "what about college timing",
      "is college open on saturday",
      "tell something about college timing",
      "what is the college  hours",
      "when should i come to college",
      "when should i attend college",
      "what is my college time",
      "college timing",
      "timing college"
   ],
   "responses": [
      "College is open 8am-5pm Monday-Saturday!"
   ],
   "context_set": ""
},
 {
   "tag": "number",
   "patterns": [
      "more info",
      "contact info",
      "how to contact college",
      "college telephone number",
      "college number",
      "What is your contact no",
      "Contact number?",
      "how to call you",
      "College phone no?",
      "how can i contact you",
      "Can i get your phone number",
      "how can i call you",
      "phone number",
      "phone no",
      "call"
   ],
   "responses": [
      "You can contact at: 123456789"
   ],
   "context_set": ""
},
 {
   "tag": "course",
   "patterns": [
      "list of courses",
      "list of courses offered",
      "list of courses offered in",
      "what are the courses offered in your college?",
      "courses?",
      "courses offered",
      "courses offered in (your univrsity(UNI) name)",
      "courses you offer",
      "branches?",
      "courses available at UNI?",
      "branches available at your college?",
      "what are the courses in UNI?",
      "what are branches in UNI?",
      "what are courses in UNI?",
      "branches available in UNI?",
      "can you tell me the courses available in UNI?",
      "can you tell me the branches available in UNI?",
      "computer engineering?",
      "computer",
      "Computer engineering?",
      "it",
      "IT",
      "Information Technology",
      "AI/Ml",
      "Mechanical engineering",
      "Chemical engineering",
      "Civil engineering"
   ],
   "responses": [
      "Our university offers Information Technology, computer Engineering, Mechanical engineering,Chemical engineering, Civil engineering and extc Engineering."
   ],
   "context_set": ""
},
 {
   "tag": "fees",
   "patterns": [
      "information about fee",
      "information on fee",
      "tell me the fee",
      "college fee",
      "fee per semester",
      "what is the fee of each semester",
      "what is the fees of each year",
      "what is fee",
      "what is the fees",
      "how much is the fees",
      "fees for first year",
      "fees",
      "about the fees",
      "tell me something about the fees",
      "What is the fees of hostel",
      "how much is the fees",
      "hostel fees",
      "fees for AC room",
      "fees for non-AC room",
      "fees for Ac room for girls",
      "fees for non-Ac room for girls",
      "fees for Ac room for boys",
      "fees for non-Ac room for boys"
   ],
   "responses": [
      "For Fee detail visit <a target=\"_blank\" href=\"LINK\"> here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "location",
   "patterns": [
      "where is the college located",
      "college is located at",
      "where is college",
      "where is college located",
      "address of college",
      "how to reach college",
      "college location",
      "college address",
      "wheres the college",
      "how can I reach college",
      "whats is the college address",
      "what is the address of college",
      "address",
      "location"
   ],
   "responses": [
      "<a target=\"_blank\" href=\"ADD YOU GOOGLE MAP LINK HERE\"> here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "hostel",
   "patterns": [
      "hostel facility",
      "hostel servive",
      "hostel location",
      "hostel address",
      "hostel facilities",
      "hostel fees",
      "Does college provide hostel",
      "Is there any hostel",
      "Where is hostel",
      "do you have hostel",
      "do you guys have hostel",
      "hostel",
      "hostel capacity",
      "what is the hostel fee",
      "how to get in hostel",
      "what is the hostel address",
      "how far is hostel from college",
      "hostel college distance",
      "where is the hostel",
      "how big is the hostel",
      "distance between college and hostel",
      "distance between hostel and college"
   ],
   "responses": [
      "For hostel detail visit <a target=\"_blank\" href=\"ADD YOUR HOSTEL DETAIL PDF LINK OR ANY INFORMATION LINK OR ADD YOU OWN ANSWERS\"> here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "event",
   "patterns": [
      "events organised",
      "list of events",
      "list of events organised in college",
      "list of events conducted in college",
      "What events are conducted in college",
      "Are there any event held at college",
      "Events?",
      "functions",
      "what are the events",
      "tell me about events",
      "what about events"
   ],
   "responses": [
      "For event detail visit <a target=\"_blank\" href=\"ADD YOUR FUNCTIONS LINK OR YOUR OWN RESPONSE\"> here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "document",
   "patterns": [
      "document to bring",
      "documents needed for admision",
      "documents needed at the time of admission",
      "documents needed during admission",
      "documents required for admision",
      "documents required at the time of admission",
      "documents required during admission",
      "What document are required for admission",
      "Which document to bring for admission",
      "documents",
      "what documents do i need",
      "what documents do I need for admission",
      "documents needed"
   ],
   "responses": [
      "To know more about document required visit <a target=\"_blank\" href=\"ADD LINK OF ADMISSION GUIDANCE DOCUMENT FROM YOUR UNIVERSITY WEBSITE\"> here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "floors",
   "patterns": [
      "size of campus",
      "building size",
      "How many floors does college have",
      "floors in college",
      "floors in college",
      "how tall is UNI's College of Engineering college building",
      "floors"
   ],
   "responses": [
      "My College has total 2 floors "
   ],
   "context_set": ""
},
 {
   "tag": "syllabus",
   "patterns": [
      "Syllabus for IT",
      "what is the Information Technology syllabus",
      "syllabus",
      "timetable",
      "what is IT syllabus",
      "syllabus",
      "What is next lecture"
   ],
   "responses": [
      "Timetable provide direct to the students OR To know about syllabus visit <a target=\"_blank\" href=\"TIMETABLE LINK\"> here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "library",
   "patterns": [
      "is there any library",
      "library facility",
      "library facilities",
      "do you have library",
      "does the college have library facility",
      "college library",
      "where can i get books",
      "book facility",
      "Where is library",
      "Library",
      "Library information",
      "Library books information",
      "Tell me about library",
      "how many libraries"
   ],
   "responses": [
      "There is one huge and spacious library.timings are 8am to 6pm and for more visit <a target=\"blank\" href=\"ADD LIBRARY DETAIL LINK\">here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "infrastructure",
   "patterns": [
      "how is college infrastructure",
      "infrastructure",
      "college infrastructure"
   ],
   "responses": [
      "Our University has Excellent Infrastructure. Campus is clean. Good IT Labs With Good Speed of Internet connection"
   ],
   "context_set": ""
},
 {
   "tag": "canteen",
   "patterns": [
      "food facilities",
      "canteen facilities",
      "canteen facility",
      "is there any canteen",
      "Is there a cafetaria in college",
      "Does college have canteen",
      "Where is canteen",
      "where is cafetaria",
      "canteen",
      "Food",
      "Cafetaria"
   ],
   "responses": [
      "Our university has canteen with variety of food available"
   ],
   "context_set": ""
},
 {
   "tag": "menu",
   "patterns": [
      "food menu",
      "food in canteen",
      "Whats there on menu",
      "what is available in college canteen",
      "what foods can we get in college canteen",
      "food variety",
      "What is there to eat?"
   ],
   "responses": [
      "we serve Franky, Locho, Alu-puri, Kachori, Khavsa, Thaali and many more on menu"
   ],
   "context_set": ""
},
 {
   "tag": "placement",
   "patterns": [
      "What is college placement",
      "Which companies visit in college",
      "What is average package",
      "companies visit",
      "package",
      "About placement",
      "placement",
      "recruitment",
      "companies"
   ],
   "responses": [
      "To know about placement visit <a target=\"_blank\" href=\"PLACEMENT INFORMATION LINK FROM YOUR UNIVERSITY WEBSITE IF THEY HAVE\">here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "ithod",
   "patterns": [
      "Who is HOD",
      "Where is HOD",
      "it hod",
      "name of it hod"
   ],
   "responses": [
      "All engineering departments have only one hod XYZ who available on (Place name)"
   ],
   "context_set": ""
},
 {
   "tag": "computerhod",
   "patterns": [
      "Who is computer HOD",
      "Where is computer HOD",
      "computer hod",
      "name of computer hod"
   ],
   "responses": [
      "All engineering departments have only one hod XYZ who available on (PLACE NAME)"
   ],
   "context_set": ""
},
 {
   "tag": "extchod",
   "patterns": [
      "Who is extc HOD",
      "Where is  extc HOD",
      "extc hod",
      "name of extc hod"
   ],
   "responses": [
      "Different school wise hod are different.So be more clear with your school or department"
   ],
   "context_set": ""
},
 {
   "tag": "principal",
   "patterns": [
      "what is the name of principal",
      "whatv is the principal name",
      "principal name",
      "Who is college principal",
      "Where is principal's office",
      "principal",
      "name of principal"
   ],
   "responses": [
      "XYZ is college principal and if you need any help then call your branch hod first.That is more appropriate"
   ],
   "context_set": ""
},
 {
   "tag": "sem",
   "patterns": [
      "exam dates",
      "exam schedule",
      "When is semester exam",
      "Semester exam timetable",
      "sem",
      "semester",
      "exam",
      "when is exam",
      "exam timetable",
      "exam dates",
      "when is semester"
   ],
   "responses": [
      "Here is the Academic Calendar  <a target=\"_blank\" href=\"YOUR ACADEMIC CALENDER\">website</a>"
   ],
   "context_set": ""
},
 {
   "tag": "admission",
   "patterns": [
      "what is the process of admission",
      "what is the admission process",
      "How to take admission in your college",
      "What is the process for admission",
      "admission",
      "admission process"
   ],
   "responses": [
      "Application can also be submitted online through the Unversity's  <a target=\"_blank\" href=\"LINK OF ADMISSION DOCUMENT\">website</a>"
   ],
   "context_set": ""
},
 {
   "tag": "scholarship",
   "patterns": [
      "scholarship",
      "Is scholarship available",
      "scholarship engineering",
      "scholarship it",
      "scholarship ce",
      "scholarship mechanical",
      "scholarship civil",
      "scholarship chemical",
      "scholarship for AI/ML",
      "available scholarships",
      "scholarship for computer engineering",
      "scholarship for IT engineering",
      "scholarship for mechanical engineering",
      "scholarship for civil engineering",
      "scholarship for chemical engineering",
      "list of scholarship",
      "comps scholarship",
      "IT scholarship",
      "mechanical scholarship",
      "civil scholarship",
      "chemical scholarship",
      "automobile scholarship",
      "first year scholarship",
      "second year scholarship",
      "third year scholarship",
      "fourth year scholarship"
   ],
   "responses": [
      "Many government scholarships are supported by our university. For details and updates visit <a target=\"_blank\" href=\"(SCHOLARSHIP DETAILS LINK)\">here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "facilities",
   "patterns": [
      "What facilities college provide",
      "College facility",
      "What are college facilities",
      "facilities",
      "facilities provided"
   ],
   "responses": [
      "Our university's Engineering department provides fully AC Lab with internet connection, smart classroom, Auditorium, library,canteen"
   ],
   "context_set": ""
},
 {
   "tag": "college intake",
   "patterns": [
      "max number of students",
      "number of seats per branch",
      "number of seats in each branch",
      "maximum number of seats",
      "maximum students intake",
      "What is college intake",
      "how many stundent are taken in each branch",
      "seat allotment",
      "seats"
   ],
   "responses": [
      "For IT, Computer and extc 60 per branch and seat may be differ for different department."
   ],
   "context_set": ""
},
 {
   "tag": "uniform",
   "patterns": [
      "college dress code",
      "college dresscode",
      "what is the uniform",
      "can we wear casuals",
      "Does college have an uniform",
      "Is there any uniform",
      "uniform",
      "what about uniform",
      "do we have to wear uniform"
   ],
   "responses": [
      "ENTER YOUR OWN UNIVERSITY UNIFORM CIRCULER"
   ],
   "context_set": ""
},
 {
   "tag": "committee",
   "patterns": [
      "what are the different committe in college",
      "different committee in college",
      "Are there any committee in college",
      "Give me committee details",
      "committee",
      "how many committee are there in college"
   ],
   "responses": [
      "For the various committe in college contact this number: ADD NUMBER"
   ],
   "context_set": ""
},
 {
   "tag": "random",
   "patterns": [
      "I love you",
      "Will you marry me",
      "Do you love me"
   ],
   "responses": [
      "I am not program for this, please ask appropriate query"
   ],
   "context_set": ""
},
 {
   "tag": "swear",
   "patterns": [
      "fuck",
      "bitch",
      "shut up",
      "hell",
      "stupid",
      "idiot",
      "dumb ass",
      "asshole",
      "fucker"
   ],
   "responses": [
      "please use appropriate language",
      "Maintaining decency would be appreciated"
   ],
   "context_set": ""
},
 {
   "tag": "vacation",
   "patterns": [
      "holidays",
      "when will semester starts",
      "when will semester end",
      "when is the holidays",
      "list of holidays",
      "Holiday in these year",
      "holiday list",
      "about vacations",
      "about holidays",
      "When is vacation",
      "When is holidays",
      "how long will be the vacation"
   ],
   "responses": [
      "Academic calender is given to you by your class-soordinators after you join your respective classes"
   ],
   "context_set": ""
},
 {
   "tag": "sports",
   "patterns": [
      "sports and games",
      "give sports details",
      "sports infrastructure",
      "sports facilities",
      "information about sports",
      "Sports activities",
      "please provide sports and games information"
   ],
   "responses": [
      "Our university encourages all-round development of students and hence provides sports facilities in the campus. For more details visit<a target=\"_blank\" href=/\"(LINK IF HAVE)\">here</a>"
   ],
   "context_set": ""
},
 {
   "tag": "salutaion",
   "patterns": [
      "okk",
      "okie",
      "nice work",
      "well done",
      "good job",
      "thanks for the help",
      "Thank You",
      "its ok",
      "Thanks",
      "Good work",
      "k",
      "ok",
      "okay"
   ],
   "responses": [
      "I am glad I helped you",
      "welcome, anything else i can assist you with?"
   ],
   "context_set": ""
},
 {
   "tag": "task",
   "patterns": [
      "what can you do",
      "what are the thing you can do",
      "things you can do",
      "what can u do for me",
      "how u can help me",
      "why i should use you"
   ],
   "responses": [
      "I can answer to low-intermediate questions regarding college",
      "You can ask me questions regarding college, and i will try to answer them"
   ],
   "context_set": ""
},
 {
   "tag": "ragging",
   "patterns": [
      "ragging",
      "is ragging practice active in college",
      "does college have any antiragging facility",
      "is there any ragging cases",
      "is ragging done here",
      "ragging against",
      "antiragging facility",
      "ragging juniors",
      "ragging history",
      "ragging incidents"
   ],
   "responses": [
      "We are Proud to tell you that our college provides ragging free environment, and we have strict rules against ragging"
   ],
   "context_set": ""
},
 {
   "tag": "hod",
   "patterns": [
      "hod",
      "hod name",
      "who is the hod"
   ],
   "responses": [
      "HODs differ for each branch, please be more specific like: (HOD it)"
   ],
   "context_set": ""
}
]
}


with open('intents.json', 'w') as file:
    json.dump(intent, file)


# In[3]:


nltk.download('stopwords')
nltk.download('punkt')

with open('intents.json') as file:
    intents = json.load(file)


# In[4]:


text_data = []
labels = []
stopwords = set(nltk.corpus.stopwords.words('english'))
for intent in intents['intents']:
    for example in intent['patterns']:
        tokens = nltk.word_tokenize(example.lower())
        text_data.append(' '.join([token for token in tokens if token not in stopwords and token.isalpha()]))
        labels.append(intent['tag'])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)
y = labels


# In[5]:


def find_best_model(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=100000)),
        ('Multinomial Naive Bayes', MultinomialNB(alpha=0.1)),
        ('Linear SVC', LinearSVC(C=1, max_iter=100000)),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f'{name}: {score:.4f}')

 
    best_model = max(models, key=lambda x: x[1].score(X_test, y_test))
    print(f'\nBest model: {best_model[0]}')
    return best_model[1]


# In[6]:


best_model = find_best_model(X, y)


# In[7]:


def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]
    
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            return response


# In[8]:


print('Hello! I am a chatbot. How can I help you today? Type "quit" to exit.')
while True:
    user_input = input('> ')
    if user_input.lower() == 'quit':
        break
    response = chatbot_response(user_input)
    print(response)


# In[9]:


import pickle


with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('intents.json', 'w') as f:
    json.dump(intents, f)


# In[10]:


import os
import nltk
import ssl
import streamlit as st

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))


# In[11]:


counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot_response(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()


# In[ ]:




