import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class FlakeSelector:
    def __init__(self, image_path):
        self.raw_img = cv2.imread(image_path)
        if self.raw_img is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        self.display_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        
        # Interaction state
        self.pts = []          
        self.lines = []        
        self.refined_contour = None
        
        self.setup_ui()

    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.subplots_adjust(bottom=0.2) 
        self.ax.imshow(self.display_img)
        self.ax.set_title("Trace the flake boundary | Confirm -> Reset -> Exit")

        # Create Buttons
        ax_confirm = plt.axes([0.3, 0.05, 0.1, 0.06])
        ax_reset = plt.axes([0.45, 0.05, 0.1, 0.06])
        ax_exit = plt.axes([0.6, 0.05, 0.1, 0.06])

        self.btn_confirm = Button(ax_confirm, 'Confirm', color='lightgreen')
        self.btn_reset = Button(ax_reset, 'Reset', color='lightyellow')
        self.btn_exit = Button(ax_exit, 'Exit', color='tomato')

        # Bind Events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.btn_confirm.on_clicked(self.confirm_selection)
        self.btn_reset.on_clicked(self.reset_selection)
        self.btn_exit.on_clicked(self.exit_app)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        new_pt = (event.xdata, event.ydata)
        self.pts.append(new_pt)
        if len(self.pts) > 1:
            line, = self.ax.plot([self.pts[-2][0], self.pts[-1][0]], 
                                [self.pts[-2][1], self.pts[-1][1]], 
                                color='red', lw=2)
            self.lines.append(line)
        self.fig.canvas.draw()

    def reset_selection(self, event):
        self.pts = []
        for line in self.lines: line.remove()
        self.lines = []
        self.fig.canvas.draw()

    def confirm_selection(self, event):
        if len(self.pts) < 3: return
        # Close the visual loop
        line, = self.ax.plot([self.pts[-1][0], self.pts[0][0]], 
                            [self.pts[-1][1], self.pts[0][1]], color='red', lw=2)
        self.lines.append(line)
        self.fig.canvas.draw()
        self.refine_logic()

    def refine_logic(self):
        # 1. Create mask
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        poly_pts = np.array(self.pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], 255)

        # 2. Local Enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        local_enhanced = clahe.apply(self.gray)

        # 3. Thresholding
        masked_pixels = local_enhanced[mask > 0]
        if masked_pixels.size == 0: return

        ret, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = float(np.max(ret)) * 0.7

        _, refined_roi = cv2.threshold(local_enhanced, thresh_val, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.bitwise_and(refined_roi, refined_roi, mask=mask)

        # 4. Contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        if contours:
            self.refined_contour = max(contours, key=cv2.contourArea)
            display_copy = self.display_img.copy()
            cv2.drawContours(display_copy, [self.refined_contour], -1, (255, 255, 0), 2)
            self.ax.clear()
            self.ax.imshow(display_copy)
            self.ax.set_title("Edge Refined (Yellow)! Click Exit to proceed.")
            self.fig.canvas.draw()

    def get_refined_roi(self, padding=60):
        if self.refined_contour is None:
            return None, None
        
        # Using OpenCV to Find the Minimum Bounding Rectangle of a Contour
        x, y, w, h = cv2.boundingRect(self.refined_contour)
        img_h, img_w = self.gray.shape
        
        # Calculate ROI coordinates
        y1, y2 = max(0, y-padding), min(img_h, y+h+padding)
        x1, x2 = max(0, x-padding), min(img_w, x+w+padding)
        
        roi_color = self.display_img[y1:y2, x1:x2]
        roi_gray = self.gray[y1:y2, x1:x2]
        return roi_color, roi_gray

    def exit_app(self, event):
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.refined_contour

class EdgeDetector:

    def __init__(self, roi_img, gray_roi):

        self.roi_img = roi_img
        self.gray_roi = gray_roi

        # Candidate lines
        self.candidate_lines = []

        # Hover line object
        self.hover_line_obj = None

        # Selected features
        self.classified_features = []

        # Manual drawing state
        self.manual_mode = False
        self.manual_pts = []

        # Current category
        self.current_mode = 0

        self.modes = [
            {"title": "Select Folding Edges (Red)", "color": "red"},
            {"title": "Select Straight Edges (Green)", "color": "lime"},
            {"title": "Select Wrinkles (Blue)", "color": "cyan"}
        ]

        self.detect_candidates()

    # ============================================================
    # Detect candidate edges
    # ============================================================

    def detect_candidates(self):

        # Slight smoothing
        blurred = cv2.GaussianBlur(
            self.gray_roi,
            (3, 3),
            0
        )

        # Edge detection
        edges = cv2.Canny(
            blurred,
            30,
            50
        )

        # Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=10,
            maxLineGap=100
        )

        if lines is not None:

            raw_lines = [line[0] for line in lines]

            self.candidate_lines = self.merge_similar_lines(
                raw_lines
            )

            print(
                f"Detected "
                f"{len(self.candidate_lines)} "
                f"candidate edges."
            )

        else:

            self.candidate_lines = []

    # ============================================================
    # Merge similar lines
    # ============================================================

    def merge_similar_lines(self, lines):

        if not lines:
            return []

        return lines

    # ============================================================
    # Find closest candidate line
    # ============================================================

    def get_closest_line(
        self,
        mouse_x,
        mouse_y,
        threshold=25
    ):

        best_line = None
        min_dist = threshold

        p3 = np.array([mouse_x, mouse_y])

        for line in self.candidate_lines:

            p1 = np.array([line[0], line[1]])
            p2 = np.array([line[2], line[3]])

            line_vec = p2 - p1

            line_len = np.linalg.norm(line_vec)

            if line_len == 0:
                continue

            dist = np.abs(
                np.cross(line_vec, p1 - p3)
            ) / line_len

            t = np.dot(
                p3 - p1,
                line_vec
            ) / (line_len ** 2)

            if 0 <= t <= 1:

                if dist < min_dist:

                    min_dist = dist
                    best_line = line

            elif -0.1 <= t <= 1.1:

                if dist < min_dist * 0.8:

                    min_dist = dist
                    best_line = line

        return best_line

    # ============================================================
    # Mouse hover
    # ============================================================

    def on_mouse_move(self, event):

        if event.inaxes != self.ax:
            return

        # Disable hover in manual mode
        if self.manual_mode:
            return

        target_line = self.get_closest_line(
            event.xdata,
            event.ydata
        )

        # Remove previous hover
        if self.hover_line_obj:

            self.hover_line_obj.remove()
            self.hover_line_obj = None

        # Draw hover line
        if target_line is not None:

            x1, y1, x2, y2 = target_line

            self.hover_line_obj, = self.ax.plot(
                [x1, x2],
                [y1, y2],
                color='white',
                lw=4,
                alpha=0.6,
                ls='--'
            )

        self.fig.canvas.draw_idle()

    # ============================================================
    # Mouse click
    # ============================================================

    def on_click(self, event):

        if event.inaxes != self.ax:
            return

        current_color = self.modes[
            self.current_mode
        ]["color"]

        # ========================================================
        # Manual mode
        # ========================================================

        if self.manual_mode:

            self.manual_pts.append(
                (event.xdata, event.ydata)
            )

            print(
                f"Manual point "
                f"{len(self.manual_pts)} selected."
            )

            # Two points define one line
            if len(self.manual_pts) == 2:

                x1, y1 = self.manual_pts[0]
                x2, y2 = self.manual_pts[1]

                line_coords = [
                    x1,
                    y1,
                    x2,
                    y2
                ]

                ln_obj, = self.ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=current_color,
                    lw=3
                )

                self.classified_features.append({
                    "mode": self.current_mode,
                    "coords": line_coords,
                    "obj": ln_obj,
                    "source": "manual"
                })

                print("Manual edge added.")

                self.manual_pts = []

                self.fig.canvas.draw()

            return

        # ========================================================
        # Auto-detected mode
        # ========================================================

        target_line = self.get_closest_line(
            event.xdata,
            event.ydata
        )

        if target_line is not None:

            x1, y1, x2, y2 = target_line

            ln_obj, = self.ax.plot(
                [x1, x2],
                [y1, y2],
                color=current_color,
                lw=3
            )

            self.classified_features.append({
                "mode": self.current_mode,
                "coords": target_line,
                "obj": ln_obj,
                "source": "auto"
            })

            self.fig.canvas.draw()

    # ============================================================
    # Toggle manual mode
    # ============================================================

    def toggle_manual_mode(self, event):

        self.manual_mode = not self.manual_mode

        if self.manual_mode:

            print("\nManual mode ON")

            self.ax.set_title(
                self.modes[self.current_mode]["title"]
                + " | MANUAL MODE"
            )

        else:

            print("\nManual mode OFF")

            self.manual_pts = []

            self.ax.set_title(
                self.modes[self.current_mode]["title"]
            )

        self.fig.canvas.draw()

    # ============================================================
    # Confirm current category
    # ============================================================

    def confirm_step(self, event):

        # Exit manual mode automatically
        self.manual_mode = False
        self.manual_pts = []

        # Next category
        if self.current_mode < 2:

            self.current_mode += 1

            self.ax.set_title(
                self.modes[self.current_mode]["title"]
            )

            print(
                f"\nSwitched to:\n"
                f"{self.modes[self.current_mode]['title']}"
            )

            self.fig.canvas.draw()

        else:

            print("\nSelection complete.")

            plt.close(self.fig)

    # ============================================================
    # Reset current category
    # ============================================================

    def reset_step(self, event):

        new_list = []

        for feat in self.classified_features:

            if feat["mode"] == self.current_mode:

                feat["obj"].remove()

            else:

                new_list.append(feat)

        self.classified_features = new_list

        self.manual_pts = []

        self.fig.canvas.draw()

    # ============================================================
    # Run GUI
    # ============================================================

    def run(self):

        self.fig, self.ax = plt.subplots(
            figsize=(10, 8)
        )

        self.fig.subplots_adjust(bottom=0.2)

        self.ax.imshow(self.roi_img)

        self.ax.set_title(
            self.modes[self.current_mode]["title"]
        )

        # ========================================================
        # Buttons
        # ========================================================

        ax_confirm = plt.axes(
            [0.18, 0.05, 0.16, 0.06]
        )

        ax_reset = plt.axes(
            [0.42, 0.05, 0.16, 0.06]
        )

        ax_manual = plt.axes(
            [0.66, 0.05, 0.18, 0.06]
        )

        self.btn_confirm = Button(
            ax_confirm,
            'Confirm',
            color='lightgreen'
        )

        self.btn_reset = Button(
            ax_reset,
            'Reset',
            color='lightyellow'
        )

        self.btn_manual = Button(
            ax_manual,
            'Manual Mode',
            color='lightblue'
        )

        # ========================================================
        # Register events
        # ========================================================

        self.fig.canvas.mpl_connect(
            'motion_notify_event',
            self.on_mouse_move
        )

        self.fig.canvas.mpl_connect(
            'button_press_event',
            self.on_click
        )

        self.btn_confirm.on_clicked(
            self.confirm_step
        )

        self.btn_reset.on_clicked(
            self.reset_step
        )

        self.btn_manual.on_clicked(
            self.toggle_manual_mode
        )

        plt.show()

        # ========================================================
        # Organize output
        # ========================================================

        results = {
            "folding": [],
            "straight": [],
            "wrinkle": []
        }

        mapping = {
            0: "folding",
            1: "straight",
            2: "wrinkle"
        }

        for feat in self.classified_features:

            results[
                mapping[feat["mode"]]
            ].append(
                feat["coords"]
            )

        return results
    
class AngleCalculator:
    def __init__(self, feature_data):
        """
        feature_data:
        {
            'folding':  [line1, line2, ...],
            'straight': [line1, line2, ...],
            'wrinkle':  [line1, line2, ...]
        }

        Every line:
        [x1, y1, x2, y2]
        """
        self.features = feature_data
        self.results = {
            'straight': [],
            'wrinkle': []
        }

    def calculate_angle(self, line1, line2):
        """
        Compute the angle between two line segments.
        """
        v1 = np.array([
            line1[2] - line1[0],
            line1[3] - line1[1]
        ])

        v2 = np.array([
            line2[2] - line2[0],
            line2[3] - line2[1]
        ])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return None

        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)

        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine_angle))

        # Convert to acute angle
        if angle > 90:
            angle = 180 - angle

        return round(angle, 2)

    def run(self):

        folding_lines = self.features.get('folding', [])

        if len(folding_lines) == 0:
            print("\n[Error] No folding lines selected.")
            return None

        print("\n" + "=" * 70)

        # Table header
        header = f"{'Type':<16} | {'ID':<4}"

        for j in range(len(folding_lines)):
            header += f"| Fold-{j+1:<6}"

        print(header)
        print("-" * 70)

        # Process straight + wrinkle
        categories = [
            ('straight', 'Straight Edge'),
            ('wrinkle', 'Wrinkle')
        ]

        for key, label in categories:

            target_lines = self.features.get(key, [])

            for i, target_line in enumerate(target_lines):

                row = f"{label:<16} | {i+1:<4}"

                for folding_line in folding_lines:

                    angle = self.calculate_angle(
                        folding_line,
                        target_line
                    )

                    if angle is None:
                        row += f"| {'N/A':<7}"
                    else:
                        row += f"| {angle:<5}°  "

                print(row)

        print("=" * 70 + "\n")