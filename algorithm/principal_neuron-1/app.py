from flask import Flask, render_template, jsonify
from src.neuron_animation_generator import NeuronActivityAnimator # Adjusted import path
import os

app = Flask(__name__)

# Determine the correct path to the data file
# Assuming app.py is in the root and data is in data/EMtrace01-3标签版.csv
# Adjust if your structure is different or if you want to make it configurable
current_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_FILE_PATH = os.path.join(current_dir, 'data', 'EMtrace01-3标签版.csv')
DATA_FILE_PATH = os.path.join(current_dir, 'data', 'processed_EMtrace_simplified.csv') # Updated path
POSITIONS_FILE_PATH = os.path.join(current_dir, 'data', 'EMtrace01_Max_position.csv') # Path to your positions file

# Initialize the animator. This will be created once when the app starts.
# For larger applications, you might initialize it per request or use a more sophisticated setup.
try:
    # Pass both data_path and positions_path to the animator
    animator = NeuronActivityAnimator(data_path=DATA_FILE_PATH, positions_path=POSITIONS_FILE_PATH)
    animator.load_data() # Pre-load data on startup
    animator._initialize_neuron_positions() # Pre-initialize positions
except FileNotFoundError as fnf_error:
    print(f"ERROR: Data or Positions file not found. Please check paths. Details: {fnf_error}")
    animator = None
except Exception as e:
    print(f"Error initializing NeuronActivityAnimator: {e}")
    animator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/neuron_data')
def get_neuron_data():
    if animator:
        try:
            data = animator.get_data_for_web()
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Animator not initialized. Check server logs for file issues."}), 500

if __name__ == '__main__':
    app.run(debug=True) 