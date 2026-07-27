import os
import tkinter as tk
from tkinter import filedialog
# Ensure these classes are in flake_selector.py
from Flake_selector import FlakeSelector, EdgeDetector, AngleCalculator

def run_test():
    # 1. Initialize Tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()

    # 2. Select local image
    file_path = filedialog.askopenfilename(
        title="Select one optical image of 2D flake",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )

    if not file_path:
        print("No file selected; the program is exiting.")
        return

    print(f"--- Loading the image ---: {file_path}")
    
    try:
        # --- STAGE 1: Flake Selection & Edge Refinement ---
        selector = FlakeSelector(file_path)
        refined_cnt = selector.run()

        if refined_cnt is None:
            print("No flake was selected or refined. Exiting.")
            return

        # --- STAGE 2: ROI Extraction & Feature Classification ---
        print("Extracting ROI and starting Stage 2 Analysis...")
        
        roi_color, roi_gray = selector.get_refined_roi(padding=60)
        
        if roi_color is not None:
            analyzer = EdgeDetector(roi_color, roi_gray)
            selected_features = analyzer.run()
            
            print("\n" + "="*30)
            print("FINAL SELECTION RESULTS:")
            print(f"Folding Edges:  {len(selected_features['folding'])} identified")
            print(f"Straight Edges: {len(selected_features['straight'])} identified")
            print(f"Wrinkles:       {len(selected_features['wrinkle'])} identified")
            print("="*30)
            
            # --- STAGE 3: Angle Calculation ---
            if selected_features['folding']:
                print("\nInitiating Stage 3: Angle Calculation relative to folding edge...")
                calculator = AngleCalculator(selected_features)
                angle_results = calculator.run()
            else:
                print("\n[Skip Stage 3] No folding edge selected. Angle calculation requires a 0° reference.")
                
        else:
            print("Failed to extract ROI.")

    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()