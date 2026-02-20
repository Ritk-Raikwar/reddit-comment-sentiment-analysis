import matplotlib
matplotlib.use('Agg') # Non-interactive backend

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import sys
import os
from src.pipeline.prediction_pipeline import PredictionPipeline # pylance error de ra tha ki from prediction_pipeline se seedha import nhi kar sakte root mei src h woh root mei pipeline find karra toh .vscode folder mei setting.json mei ./src

app = Flask(__name__)
CORS(app) 

# --- 2. Initialize Pipeline (Happens ONCE when server starts) ---
print("Initializing Prediction Pipeline...")
pipeline = PredictionPipeline()
print("Pipeline Ready!")

@app.route('/')
def home():
    return "YouTube Sentiment Analysis API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Use the pipeline to predict
        predictions = pipeline.predict(comments)
        
        # Zip results together
        response = [
            {"comment": c, "sentiment": p} 
            for c, p in zip(comments, predictions)
        ]
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments') # Expecting list of {text: "...", timestamp: "..."}
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        raw_texts = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Use Pipeline
        predictions = pipeline.predict(raw_texts)
        
        # Return structured response
        response = [
            {"comment": txt, "sentiment": pred, "timestamp": ts} 
            for txt, pred, ts in zip(raw_texts, predictions, timestamps)
        ]
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Use Pipeline to get clean text string
        text = pipeline.get_preprocessed_text(comments)

        if not text.strip():
             return jsonify({"error": "No valid text after preprocessing"}), 400

        # Generate WordCloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='black',
            colormap='Blues',
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        
        if sum(sizes) == 0:
            return jsonify({"error": "Sum of sentiments is 0"}), 400
            
        # Brighter, modern colors for dark mode
        colors = ['#4CAF50', '#9E9E9E', '#F44336'] 

        # Explicitly set text color to white
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=140, 
            textprops={'color': 'white', 'weight': 'bold'} # <-- THE FIX
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        
        if monthly_counts.empty:
             return jsonify({"error": "Not enough data for trend graph"}), 400

        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns: monthly_percentages[val] = 0
            
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # --- DARK MODE STYLING FOR PLOT ---
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {-1: '#F44336', 0: '#9E9E9E', 1: '#4CAF50'}

        for val in [-1, 0, 1]:
            ax.plot(monthly_percentages.index, monthly_percentages[val], 
                     marker='o', label=sentiment_labels[val], color=colors[val], linewidth=2)

        ax.set_title('Sentiment Trend Over Time', color='white', weight='bold')
        ax.set_ylabel('Percentage %', color='white')
        
        # Make axes and ticks white
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
            
        ax.grid(True, color='#444444', linestyle='--') # Subtle dark grid
        
        # Style the legend for dark mode
        legend = ax.legend(facecolor='#222222', edgecolor='#444444')
        for text in legend.get_texts():
            text.set_color("white")

        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True) # Save with transparent background
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)