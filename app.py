from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Logistic Regression model and the corresponding TF-IDF vectorizer
with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize GPT-2 model and tokenizer
gpt2_model = TFGPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Predefined static suggestions
static_suggestions = {
    "Anxiety": [
        "Practice mindfulness and deep breathing exercises.",
        "Consider talking to a counselor or therapist.",
        "Try to limit caffeine and alcohol consumption.",
    ],
    "EatingDisorders": [
        "Consult a nutritionist to develop a healthy eating plan.",
        "Consider therapy to address underlying psychological factors.",
        "Engage in activities that focus on body positivity and self-esteem.",
    ],
    "NarcissisticAbuse": [
        "Consider talking to a therapist to process the trauma.",
        "Set healthy boundaries with toxic individuals.",
        "Seek support from loved ones or support groups.",
    ],
    "Schizophrenia": [
        "Work with a psychiatrist for medication management.",
        "Join a support group for people with schizophrenia.",
        "Develop a routine that includes healthy habits like exercise and sleep.",
    ],
    "bipolar2": [
        "Consider mood stabilizers prescribed by a healthcare provider.",
        "Practice good sleep hygiene and avoid alcohol or drugs.",
        "Engage in regular therapy sessions, such as CBT.",
    ],
    "depression": [
        "Engage in regular physical activity, like a walk or workout.",
        "Try to maintain a healthy, balanced diet.",
        "Connect with friends or family for support.",
    ],
    "mentalhealths": [
        "Take small breaks during stressful moments.",
        "Reach out to a mental health professional for guidance.",
        "Engage in self-care activities, like reading or taking walks.",
    ],
    "ocd": [
        "Consider CBT (Cognitive Behavioral Therapy) for OCD.",
        "Practice mindfulness to manage intrusive thoughts.",
        "Try to maintain a structured routine.",
    ],
    "ptsd": [
        "Consider seeking professional trauma therapy.",
        "Practice grounding techniques to stay present.",
        "Try to engage in regular physical activity to reduce stress.",
    ],
}




# Function to get top 3 predictions
def get_top_3_predictions(user_input):
    text_tfidf = vectorizer.transform([user_input])
    probs = model.predict_proba(text_tfidf)[0]
    top_3_indices = probs.argsort()[-3:][::-1]
    top_3_categories = label_encoder.inverse_transform(top_3_indices)
    top_3_probs = probs[top_3_indices]
    return top_3_categories, top_3_probs

# Function to dynamically generate suggestions using GPT-2
def generate_gpt2_suggestions(categories):
    generated_suggestions = {}
    for category in categories:
        input_text = (
            f"The user is struggling with {category}, which is affecting their mental and emotional well-being. "
            f"Offer a 3-step, practical, and actionable advice for someone dealing with {category}. "
            f"Include specific techniques, coping strategies, and emotional support they can use."
        )
        inputs = tokenizer.encode_plus(
            input_text, return_tensors='pt', padding=True, truncation=True, max_length=300
        )
        output = gpt2_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=300,
            num_beams=2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_suggestions[category] = generated_text
    return generated_suggestions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Get top 3 predictions
        top_3_categories, top_3_probs = get_top_3_predictions(user_input)

        # Generate dynamic suggestions using GPT-2
        dynamic_suggestions = generate_gpt2_suggestions(top_3_categories)

        # Prepare the response data
        response_data = {
            "top_3_categories": top_3_categories,
            "top_3_probs": top_3_probs,
            "static_suggestions": static_suggestions,
            "dynamic_suggestions": dynamic_suggestions,
        }

        return render_template('index.html', user_input=user_input, response_data=response_data)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
