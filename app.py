from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import textstat
import torch
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = False

print("🔄 Loading BART model (this takes 1-2 minutes)...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU
)
print("✅ BART model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data['text']
        max_length = data.get('max_length', 130)
        min_length = data.get('min_length', 30)

        if len(text.split()) < 10:
            return jsonify({'error': 'Text too short (min 10 words)'}), 400

        summary_result = summarizer(
            text[:1500],
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )

        summary = summary_result[0]['summary_text']
        readability_score = textstat.flesch_reading_ease(text)
        readability_label = (
            "Very Difficult" if readability_score < 30 else 
            "Difficult" if readability_score < 50 else 
            "Fairly Difficult" if readability_score < 60 else 
            "Standard" if readability_score < 70 else 
            "Fairly Easy" if readability_score < 80 else 
            "Easy" if readability_score < 90 else "Very Easy"
        )

        return jsonify({
            'summary': summary,
            'readability': f"{readability_score:.1f} ({readability_label})",
            'word_count_original': len(text.split()),
            'word_count_summary': len(summary.split())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



