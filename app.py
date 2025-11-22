from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load Models
MODEL_DIR = 'models'

def load_models():
    try:
        f_model = joblib.load(os.path.join(MODEL_DIR, 'filter_model.pkl'))
        s_model = joblib.load(os.path.join(MODEL_DIR, 'sort_model.pkl'))
        print("✅ Models loaded successfully.")
        return f_model, s_model
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        print("Did you run 'python3 train_model.py' yet?")
        return None, None

filter_model, sort_model = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Reload if None (in case user trained after starting app)
    global filter_model, sort_model
    if not filter_model:
        filter_model, sort_model = load_models()
        
    if not filter_model or not sort_model:
        return jsonify({'error': 'Models not loaded. Train them first!'}), 500

    data = request.get_json()
    tweets = data.get('tweets', [])
    results = []

    for tweet in tweets:
        # 1. FILTER STAGE
        filter_pred = filter_model.predict([tweet])[0]
        filter_label = str(filter_pred).lower()
        
        # Logic: If label is '1', 'hate', 'offensive', or 'off', we flag it.
        # Adjust this list if your specific dataset uses different codes (like '2')
        is_filtered = False
        if filter_label in ['1', 'hate', 'offensive', 'off', 'spam']: 
            is_filtered = True

        # 2. SORT STAGE
        sort_pred = sort_model.predict([tweet])[0]
        sort_label = str(sort_pred).lower()

        results.append({
            'text': tweet,
            'filtered': is_filtered,
            'filter_label': filter_label,
            'sentiment': sort_label
        })

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, port=5001)