import os
import sys
import joblib
import pandas as pd
import numpy as np

# --- 1. Dynamic Path Setup to find 'src' ---
# This allows us to import from src.utils regardless of where this script is run
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. Import the SHARED Preprocessing Logic ---
# This is the "Gold Standard" - reusing the exact code from training
from src.utils.common_text_preprocess import preprocess_text

class PredictionPipeline:
    def __init__(self):
        """
        Initialize the pipeline: Load model and vectorizer ONCE from disk.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, '..', '..'))
        
        # Paths to your trained artifacts
        self.model_path = os.path.join(self.project_root, 'models', 'lightgbm_model.pkl')
        self.vectorizer_path = os.path.join(self.project_root, 'models', 'tfidf_vectorizer.pkl')

        print(f"Loading model from: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        except FileNotFoundError:
            raise FileNotFoundError("Model or Vectorizer not found! Did you run 'dvc repro'?")

    def predict(self, comments):
        """
        Args:
            comments (list): List of raw comment strings.
        Returns:
            list: List of predictions (e.g., ['1', '-1', '0'])
        """
        if not comments:
            return []

        # 1. Preprocess (Using the shared utility!)
        # We apply the function to every comment in the list
        preprocessed = [preprocess_text(c) for c in comments]
        
        # 2. Vectorize
        vectors = self.vectorizer.transform(preprocessed)
        
        # 2.5 THE FIX: Give LightGBM its REAL word names back!
        input_df = pd.DataFrame(vectors.toarray(), columns=self.vectorizer.get_feature_names_out())
        
        # 3. Predict
        predictions = self.model.predict(input_df)
        
        # 4. Convert to strings (for JSON serialization)
        return [str(p) for p in predictions]

    def get_preprocessed_text(self, comments):
        """Helper for WordCloud generation."""
        # Reuses the same shared logic
        preprocessed = [preprocess_text(c) for c in comments]
        return ' '.join(preprocessed)
    




## simplified version jo mene pehle dekha h aur ...  
    
# import os
# import joblib
# import pandas as pd
# from src.utils.common_text_preprocess import preprocess_text

# class PredictionPipeline:
#     def __init__(self):
#         # CORRECT PATHS based on your dvc.yaml
#         self.vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
#         self.model_path = os.path.join('models', 'lightgbm_model.pkl')

#     def predict(self, text):
#         try:
#             # 1. Load Artifacts
#             # (Check if files exist first to avoid crashing)
#             if not os.path.exists(self.vectorizer_path):
#                 raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}. Run 'dvc repro' first.")
            
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"Model not found at {self.model_path}. Run 'dvc repro' first.")

#             vectorizer = joblib.load(self.vectorizer_path)
#             model = joblib.load(self.model_path)

#             # 2. Preprocess
#             clean_text = preprocess_text(text)

#             # 3. Vectorize
#             text_vectorized = vectorizer.transform([clean_text])

#             # 4. Predict
#             prediction = model.predict(text_vectorized)

#             # 5. Return Result
#             if prediction[0] == -1:
#                 return "Negative"
#             elif prediction[0] == 0:
#                 return "Neutral"
#             else:
#                 return "Positive"
        
#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             return "Error"