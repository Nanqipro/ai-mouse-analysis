import numpy as np
import pandas as pd
# Removed matplotlib imports as they are not directly needed for data preparation for web

class NeuronActivityAnimator:
    """
    Processes neuron activity data and prepares it for web visualization.
    """
    def __init__(self, data_path, positions_path=None):
        self.data_path = data_path
        self.positions_path = positions_path
        self.df = None
        self.neuron_cols = []
        self.behavior_labels = []
        self.neuron_positions = {}  # {'Neuron_X': (x, y)}
        self.normalized_activity = None
        self.global_min_activity = 0
        self.global_max_activity = 1
        self.data_loaded = False
        self.positions_generated = False

    def _load_custom_positions(self):
        """Loads neuron positions from a specified CSV file."""
        if not self.positions_path:
            print("DEBUG: No custom positions_path provided.")
            return False
        
        try:
            print(f"DEBUG: Attempting to load custom positions from {self.positions_path}")
            positions_df = pd.read_csv(self.positions_path)
            # Expected columns in positions_df: 'number', 'relative_x', 'relative_y'
            if not all(col in positions_df.columns for col in ['number', 'relative_x', 'relative_y']):
                print(f"ERROR: Position file {self.positions_path} is missing required columns ('number', 'relative_x', 'relative_y').")
                return False

            loaded_positions = {}
            for _, row in positions_df.iterrows():
                neuron_id_num = int(row['number']) # e.g., 1, 2, 3
                neuron_col_name = f"n{neuron_id_num}" # e.g., "n1", "n2"
                
                # Only load positions for neurons that exist in our activity data
                if neuron_col_name in self.neuron_cols:
                    loaded_positions[neuron_col_name] = (row['relative_x'], row['relative_y'])
                else:
                    print(f"DEBUG: Neuron ID {neuron_id_num} (column {neuron_col_name}) from positions file not found in activity data neuron columns. Skipping.")
            
            if not loaded_positions:
                print("ERROR: No positions were successfully loaded and mapped from the positions file. Check column names and neuron IDs.")
                return False

            self.neuron_positions = loaded_positions
            self.positions_generated = True # Mark as generated/loaded
            print(f"DEBUG: Successfully loaded {len(self.neuron_positions)} custom neuron positions from {self.positions_path}.")
            # print(f"DEBUG: Example custom position for n1: {self.neuron_positions.get('n1')}")
            return True
        except FileNotFoundError:
            print(f"ERROR: Positions file not found at {self.positions_path}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load or process positions from {self.positions_path}: {e}")
            return False

    def load_data(self):
        """Loads and preprocesses data with diagnostic prints."""
        if self.data_loaded:
            return

        print(f"DEBUG: Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path)
            print("DEBUG: DataFrame loaded successfully. First 5 rows:")
            print(self.df.head())
        except Exception as e:
            print(f"ERROR: Failed to load CSV: {e}")
            raise

        # Updated and simplified neuron column detection logic:
        self.neuron_cols = [col for col in self.df.columns if col.startswith('n') and col[1:].isdigit()]
        
        if not self.neuron_cols:
            print("DEBUG: No neuron columns found with 'n[digit]' pattern. Trying to infer numeric columns excluding 'stamp'.")
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            self.neuron_cols = [col for col in numeric_cols if col.lower() != 'stamp']
            if 'stamp' in self.neuron_cols and len(self.neuron_cols) > 40: # Heuristic, if 'stamp' was accidentally included
                print("DEBUG: Removing 'stamp' from inferred neuron columns.")
                self.neuron_cols = [col for col in self.neuron_cols if col.lower() != 'stamp']


        if not self.neuron_cols:
            raise ValueError("ERROR: No suitable neuron columns found (e.g., 'n1', 'n2', or other numeric). Please check data format.")
        
        print(f"DEBUG: Identified neuron columns (count: {len(self.neuron_cols)}): {self.neuron_cols[:10]}...") # Print first 10

        if 'Simplified_Label' not in self.df.columns:
            raise ValueError("ERROR: Column 'Simplified_Label' not found. Ensure data processing script ran correctly.")
        self.behavior_labels = self.df['Simplified_Label'].tolist()
        print(f"DEBUG: Behavior labels loaded. First 5: {self.behavior_labels[:5]}")
        
        activity_data = self.df[self.neuron_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Ensure all neuron columns are numeric, attempt conversion if not
        for col in self.neuron_cols:
            if not pd.api.types.is_numeric_dtype(activity_data[col]):
                print(f"DEBUG: Column {col} is not numeric. Attempting conversion.")
                try:
                    activity_data[col] = pd.to_numeric(activity_data[col], errors='coerce')
                except Exception as e:
                    print(f"ERROR: Could not convert column {col} to numeric: {e}")
                    # Decide if to raise error or fill NaNs / drop column
        
        # Handle potential NaNs from conversion or already existing
        if activity_data.isnull().values.any():
            print(f"DEBUG: NaNs found in activity data. Filling with mean of respective column.")
            for col in self.neuron_cols: # activity_data.columns might be safer if neuron_cols isn't perfect
                if activity_data[col].isnull().any():
                    activity_data[col] = activity_data[col].fillna(activity_data[col].mean())
            if activity_data.isnull().values.any(): # If still NaNs (e.g. whole column was NaN)
                 print(f"DEBUG: NaNs still present after mean fill. Filling remaining with 0.")
                 activity_data = activity_data.fillna(0)


        self.global_min_activity = activity_data.min().min()
        self.global_max_activity = activity_data.max().max()
        
        print(f"DEBUG: Global activity range: min={self.global_min_activity:.2f}, max={self.global_max_activity:.2f}")

        if self.global_max_activity == self.global_min_activity:
            # Avoid division by zero; assign a neutral value (e.g., 0.5) or handle as an error/special case.
            if self.global_max_activity == 0: # Or any other constant value
                 self.normalized_activity = pd.DataFrame(0.5, index=activity_data.index, columns=activity_data.columns)
                 print("DEBUG: Global min and max are equal (and possibly zero). Normalized activity set to 0.5.")
            else: # min == max but not zero
                 self.normalized_activity = pd.DataFrame(1.0, index=activity_data.index, columns=activity_data.columns)
                 print("DEBUG: Global min and max are equal (non-zero). Normalized activity set to 1.0 (or 0.0 if min/max was 0).")

        else:
            self.normalized_activity = (activity_data - self.global_min_activity) / \
                                       (self.global_max_activity - self.global_min_activity)
        
        print(f"DEBUG: Normalized activity data generated. Shape: {self.normalized_activity.shape}. First few rows/cols:")
        print(self.normalized_activity.iloc[:3, :3])

        self.data_loaded = True
        print(f"DEBUG: Data loaded: {len(self.neuron_cols)} neurons, {len(self.df)} behavior states.")

    def _initialize_neuron_positions(self):
        """Initializes neuron positions, trying custom CSV first, then programmatic generation."""
        if self.positions_generated: # Already done
            return

        if not self.data_loaded: # Ensure neuron_cols is populated
            self.load_data()

        custom_loaded = self._load_custom_positions()
        if custom_loaded:
            print("DEBUG: Custom positions loaded and used.")
            return # self.positions_generated is set by _load_custom_positions

        # Fallback to programmatic generation if custom positions were not loaded
        print("DEBUG: Custom positions not loaded or failed. Falling back to programmatic generation...")
        num_neurons = len(self.neuron_cols)
        if num_neurons == 0:
            print("ERROR: Cannot generate positions, no neuron columns identified for programmatic generation.")
            return

        golden_angle = np.pi * (3. - np.sqrt(5.))
        
        temp_positions = {}
        for i, neuron_name in enumerate(self.neuron_cols):
            theta = golden_angle * i 
            # Ensure radius calculation doesn't divide by zero if num_neurons is 1 after filtering, though unlikely
            radius_denominator = num_neurons if num_neurons > 0 else 1
            radius = np.sqrt((i + 1) / radius_denominator) # +1 to avoid 0 radius for first element if i starts at 0
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            temp_positions[neuron_name] = (x, y)
        self.neuron_positions = temp_positions
        self.positions_generated = True
        print(f"DEBUG: Programmatic positions generated for {num_neurons} neurons. Example for {self.neuron_cols[0] if num_neurons > 0 else 'N/A'}: {self.neuron_positions.get(self.neuron_cols[0] if num_neurons > 0 else None)}")

    def get_data_for_web(self):
        """Returns data formatted for web consumption with diagnostic prints."""
        if not self.data_loaded:
            self.load_data()
        if not self.positions_generated:
            self._initialize_neuron_positions()

        formatted_positions = []
        for neuron_id in self.neuron_cols:
            pos_tuple = self.neuron_positions.get(neuron_id)
            if pos_tuple:
                formatted_positions.append({"id": neuron_id, "x": pos_tuple[0], "y": pos_tuple[1]})
            else:
                print(f"WARNING: Position for neuron {neuron_id} not found in self.neuron_positions. It will be missing from web output.")
        
        print("DEBUG: Preparing data for web.")
        print(f"DEBUG: Number of neuron_ids (from self.neuron_cols): {len(self.neuron_cols)}")
        print(f"DEBUG: Number of neuron_positions loaded/generated (self.neuron_positions keys): {len(self.neuron_positions)}")
        print(f"DEBUG: Number of formatted_positions for web: {len(formatted_positions)}")
        print(f"DEBUG: Number of behavior_labels: {len(self.behavior_labels)}")
        print(f"DEBUG: Activity data shape (to be list of lists): {self.normalized_activity.shape if self.normalized_activity is not None else 'None'}")
        
        web_data_payload = {
            "neuron_ids": self.neuron_cols,
            "neuron_positions": formatted_positions,
            "behavior_labels": self.behavior_labels,
            "activity_data": self.normalized_activity.values.tolist() if self.normalized_activity is not None else [],
            "global_min_activity": self.global_min_activity,
            "global_max_activity": self.global_max_activity,
            "num_frames": len(self.df) if self.df is not None else 0
        }
        print(f"DEBUG: Payload for web: neuron_ids count: {len(web_data_payload['neuron_ids'])}, positions count: {len(web_data_payload['neuron_positions'])}, labels count: {len(web_data_payload['behavior_labels'])}, activity_data frames: {len(web_data_payload['activity_data'])}")
        return web_data_payload

# main() function removed as it was for GIF generation.
# If you need to test this class independently, you can add a simple test:
# if __name__ == '__main__':
#     animator = NeuronActivityAnimator(data_path='../data/EMtrace01-3标签版.csv') # Adjust path as needed
#     web_data = animator.get_data_for_web()
#     print("Data for web:")
#     print(f"Number of neurons: {len(web_data['neuron_ids'])}")
#     print(f"Number of frames: {web_data['num_frames']}")
#     # print(web_data) # Uncomment to see full data structure
 