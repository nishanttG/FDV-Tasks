import sys
import os
import csv
import random
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                               QWidget, QPushButton, QFileDialog, QMessageBox, QProgressBar)
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtCore import Qt

class LabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 3 Side Quest: Binary Labeler")
        self.resize(800, 600)
        
        # State
        self.image_folder = ""
        self.image_list = []
        self.current_index = 0
        self.labels_file = "labels.csv"
        
        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        
        # 1. Toolbar / Buttons
        self.btn_open = QPushButton("📂 Open Image Folder")
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_open.setStyleSheet("padding: 10px; font-size: 14px;")
        self.layout.addWidget(self.btn_open)
        
        # 2. Image Display
        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 2px dashed #555; background-color: #222; color: #fff;")
        self.lbl_image.setMinimumSize(400, 300)
        self.layout.addWidget(self.lbl_image)
        
        # 3. Instructions
        self.lbl_instr = QLabel("Press 'A' for Class 0 (Plane) | Press 'B' for Class 1 (Car)")
        self.lbl_instr.setAlignment(Qt.AlignCenter)
        self.lbl_instr.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        self.layout.addWidget(self.lbl_instr)
        
        # 4. Progress
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        
        # 5. Export Button
        self.btn_export = QPushButton("💾 Export Splits")
        self.btn_export.clicked.connect(self.export_splits)
        self.btn_export.setEnabled(False)
        self.layout.addWidget(self.btn_export)

        # Initialize CSV if not exists
        if not os.path.exists(self.labels_file):
            with open(self.labels_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'label'])

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder
            # Filter for images
            self.image_list = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Check what's already labeled to skip them
            labeled_imgs = self.get_labeled_images()
            self.image_list = [x for x in self.image_list if x not in labeled_imgs]
            
            if not self.image_list:
                QMessageBox.information(self, "Done", "All images in this folder are already labeled!")
                return
            
            self.current_index = 0
            self.update_ui_for_image()
            self.btn_export.setEnabled(True)

    def get_labeled_images(self):
        if not os.path.exists(self.labels_file):
            return []
        try:
            df = pd.read_csv(self.labels_file)
            return df['filename'].tolist()
        except:
            return []

    def update_ui_for_image(self):
        if self.current_index < len(self.image_list):
            filename = self.image_list[self.current_index]
            filepath = os.path.join(self.image_folder, filename)
            
            pixmap = QPixmap(filepath)
            self.lbl_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            
            self.setWindowTitle(f"Labeling: {filename}")
            self.progress.setValue(int((self.current_index / len(self.image_list)) * 100))
        else:
            self.lbl_image.setText("🎉 All Done!")
            self.lbl_image.setPixmap(QPixmap())
            self.lbl_instr.setText("Click 'Export Splits' to finish.")

    def keyPressEvent(self, event):
        if not self.image_list or self.current_index >= len(self.image_list):
            return
            
        filename = self.image_list[self.current_index]
        label = None
        
        if event.key() == Qt.Key_A:
            label = 0
        elif event.key() == Qt.Key_B:
            label = 1
            
        if label is not None:
            self.save_label(filename, label)
            self.current_index += 1
            self.update_ui_for_image()

    def save_label(self, filename, label):
        # Append to CSV
        with open(self.labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, label])
        print(f"Saved: {filename} -> {label}")

    def export_splits(self):
        # 1. Load Labels
        try:
            df = pd.read_csv(self.labels_file)
            if len(df) < 10:
                QMessageBox.warning(self, "Warning", "Label at least 10 images before splitting.")
                return
            
            # 2. Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # 3. Split (70% Train, 15% Val, 15% Test)
            n = len(df)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            df.loc[:train_end, 'split'] = 'train'
            df.loc[train_end:val_end, 'split'] = 'val'
            df.loc[val_end:, 'split'] = 'test'
            
            # 4. Save
            df.to_csv("splits.csv", index=False)
            
            # Show stats
            counts = df['split'].value_counts()
            msg = f"Exported splits.csv!\n\nTrain: {counts.get('train', 0)}\nVal: {counts.get('val', 0)}\nTest: {counts.get('test', 0)}"
            QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Dark Mode
    app.setStyle("Fusion")
    # any widgets can be the window itself
    window = LabelingApp()
    window.show()
    sys.exit(app.exec())