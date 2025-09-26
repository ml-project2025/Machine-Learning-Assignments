# src/data/splitter.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import logging


class TMNISTDataSplitter:
    """
    Splits TMNIST CSV data into train/val/test sets (70/20/10).
    Handles dataset with:
    - 'names': font name (ignored)
    - 'labels': digit label (0-9)
    - '1' to '784': pixel values (28x28 image)
    """

    def __init__(self, raw_csv_path: str, processed_dir: str, random_state: int = 42):
        self.raw_csv_path = os.path.normpath(raw_csv_path)
        self.processed_dir = os.path.normpath(processed_dir)
        self.random_state = random_state
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging to file and console"""
        log_file = os.path.join(os.path.dirname(self.raw_csv_path), "splitter.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TMNISTSplitter")

    def _create_class_dirs(self):
        """Create directory structure for train/val/test with class subfolders (0-9)"""
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.processed_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            for class_id in range(10):
                class_dir = os.path.join(split_dir, str(class_id))
                os.makedirs(class_dir, exist_ok=True)
                self.logger.debug(f"Created directory: {class_dir}")

    def _save_image(self, pixels, label, split_type, idx):
        """
        Save a single image to the correct class folder
        pixels: 784-length array of pixel values
        label: digit class (0-9)
        split_type: 'train', 'val', or 'test'
        idx: image index (for filename)
        """
        try:
            # Reshape to 28x28 and convert to uint8
            img_array = pixels.reshape(28, 28).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            # Save to correct class folder
            class_dir = os.path.join(self.processed_dir, split_type, str(label))
            img.save(os.path.join(class_dir, f"img_{idx}.png"))
        except Exception as e:
            self.logger.error(f"Failed to save image {idx} for class {label}: {str(e)}")
            raise

    def split_and_save(self):
        """
        Main method: loads data, splits, and saves images
        Returns True on success, False on error
        """
        try:
            # Validate input file
            if not os.path.exists(self.raw_csv_path):
                raise FileNotFoundError(f"Raw CSV not found at: {self.raw_csv_path}")

            # Create output directories
            self._create_class_dirs()
            self.logger.info("Created class directories")

            # Load the CSV
            self.logger.info("Loading TMNIST data...")
            df = pd.read_csv(self.raw_csv_path, low_memory=False)

            # Verify expected columns
            expected_pixel_cols = [str(i) for i in range(1, 785)]  # '1' to '784'
            if 'labels' not in df.columns or not all(col in df.columns for col in expected_pixel_cols):
                raise ValueError(
                    "CSV must contain 'labels' and pixel columns '1' through '784'. "
                    f"Found columns: {list(df.columns)[:10]}..."
                )

            # Extract features (pixels) and labels
            X = df[expected_pixel_cols].values  # Shape: (N, 784)
            y = df['labels'].values               # Shape: (N,)
            
            # Convert labels to string for folder naming
            y = y.astype(str)

            # Validate labels are digits 0-9
            unique_labels = set(y)
            invalid_labels = [lbl for lbl in unique_labels if not (lbl.isdigit() and len(lbl) == 1)]
            if invalid_labels:
                raise ValueError(f"Invalid labels found: {invalid_labels}")

            self.logger.info(f"Data loaded: {len(df)} samples, labels: {sorted(unique_labels)}")

            # Split into train (70%), temp (30%)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=0.3,
                stratify=y,
                random_state=self.random_state
            )

            # Split temp into val (20% of total) and test (10% of total)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=1/3,  # 1/3 of 30% = 10% of total
                stratify=y_temp,
                random_state=self.random_state
            )

            # Organize splits for processing
            splits = [
                (X_train, y_train, 'train'),
                (X_val, y_val, 'val'),
                (X_test, y_test, 'test')
            ]

            # Save images for each split
            for X_split, y_split, split_name in splits:
                self.logger.info(f"Saving {split_name} images ({len(X_split)} samples)...")
                for idx, (pixels, label) in enumerate(zip(X_split, y_split)):
                    self._save_image(pixels, label, split_name, idx)
                    if idx % 1000 == 0 and idx > 0:
                        self.logger.info(f"  Progress: {idx}/{len(X_split)}")

            # Final verification
            self._verify_splits()
            self.logger.info("Data split completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Critical error during data splitting: {str(e)}")
            return False

    def _verify_splits(self):
        """Verify final split ratios and class distributions"""
        self.logger.info("Verifying final split structure...")
        total_images = 0
        split_counts = {}

        for split in ['train', 'val', 'test']:
            class_counts = {}
            split_total = 0
            
            for class_id in range(10):
                class_dir = os.path.join(self.processed_dir, split, str(class_id))
                if os.path.exists(class_dir):
                    files = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    class_counts[class_id] = len(files)
                    split_total += len(files)
                else:
                    class_counts[class_id] = 0

            split_counts[split] = split_total
            total_images += split_total
            
            # Log class distribution
            dist_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
            self.logger.info(f"{split.upper()} distribution: {dist_str}")

        # Log final ratios
        if total_images > 0:
            train_ratio = split_counts['train'] / total_images
            val_ratio = split_counts['val'] / total_images
            test_ratio = split_counts['test'] / total_images
            
            self.logger.info(
                f"Final split ratios: "
                f"Train={train_ratio:.1%} ({split_counts['train']}), "
                f"Val={val_ratio:.1%} ({split_counts['val']}), "
                f"Test={test_ratio:.1%} ({split_counts['test']})"
            )


if __name__ == "__main__":
    # Get project root dynamically
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    
    # Define paths
    RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "TMNIST_Data.csv")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # Create and run splitter
    splitter = TMNISTDataSplitter(
        raw_csv_path=RAW_CSV_PATH,
        processed_dir=PROCESSED_DIR,
        random_state=42
    )
    
    success = splitter.split_and_save()
    if success:
        print("✅ Data splitting completed successfully!")
    else:
        print("❌ Data splitting failed. Check splitter.log for details.")