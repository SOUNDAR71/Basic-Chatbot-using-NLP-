import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download only 'punkt'
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Sample data
training_data = {
    "hi": "Hello! How can I help you?",
    "hello": "Hi there! How can I assist you?",
    "how are you": "I'm just a bot, but I'm doing great!",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome!",
    "what is your name": "I'm a chatbot created by Sound!",
}

X = list(training_data.keys())
y = list(training_data.values())

# Do not use custom tokenizer — use default token_pattern
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vec, y)

# Chat loop
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Bye!")
        break
    input_vec = vectorizer.transform([user_input])
    response = model.predict(input_vec)[0]
    print("Chatbot:", response)
