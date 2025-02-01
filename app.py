from flask import Flask, render_template, request
import pickle
import os
import pandas as pd 


app = Flask(__name__)

# Load the pre-trained model and vectorizer
# it pickle file path 
model_path = os.path.join('model', 'best_model.pkl')
vectorizer_path = os.path.join('model', 'tfidf_vectorizer.pkl')

print(model_path)

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        # # Transform the tweet using the TF-IDF vectorizer and ensure it's a 2D array
        # tweet_vector = vectorizer.transform([[tweet]]).toarray()  # Reshape tweet to 2D
        # Convert the tweet to a DataFrame to match the expected input format
        tweet_df = pd.DataFrame({'reviews': [review]})
        
        # Predict sentiment using the trained model
        prediction = model.predict(tweet_df)
        if prediction == 1:
           sentiment = 'Positive'
        else :
            sentiment = 'Negative'
        return render_template('result.html', tweet=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
