import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # استخدام backend لا يحتاج شاشة
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import argparse
import os
import json

from model import DigitCNN, ImprovedDigitCNN
from data_loader import MNISTDataLoader


class ModelEvaluator:
    """
    فئة لتقييم النموذج المدرب
    """
    def __init__(self, model_path, device=None):
        # اختيار الجهاز
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if major < 7:  # كرت قديم غير مدعوم
                    print("⚠️ كرت الشاشة قديم وغير مدعوم، سيتم استخدام CPU بدلًا من GPU")
                    self.device = torch.device("cpu")
                else:
                    self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """
        تحميل النموذج المدرب
        """
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # تحديد نوع النموذج
        if checkpoint['model_type'] == 'ImprovedDigitCNN':
            self.model = ImprovedDigitCNN()
        else:
            self.model = DigitCNN()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully from {self.model_path}")
        print(f"Model type: {checkpoint['model_type']}")
        print(f"Training accuracy: {checkpoint['accuracy']:.2f}%")

    def evaluate_on_test_set(self):
        """
        تقييم النموذج على مجموعة الاختبار
        """
        # تحميل بيانات الاختبار
        data_loader = MNISTDataLoader()
        _, test_loader = data_loader.get_data_loaders()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        correct = 0
        total = 0

        print("Evaluating model on test set...")

        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # الحصول على الاحتماليات والتنبؤات
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                # حفظ النتائج
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # حساب الدقة
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total

        print(f'Test Accuracy: {accuracy:.2f}%')

        return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        رسم مصفوفة الالتباس
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        os.makedirs('./saved_models', exist_ok=True)
        plt.savefig('./saved_models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("📊 Saved confusion matrix at ./saved_models/confusion_matrix.png")
        plt.close()

        return cm

    def generate_classification_report(self, y_true, y_pred):
        """
        إنشاء تقرير التصنيف المفصل
        """
        # تقرير التصنيف
        report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
        print("\nClassification Report:")
        print("=" * 60)
        print(report)

        # حساب المقاييس الإضافية
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"\nOverall Metrics:")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report
        }

    def analyze_errors(self, y_true, y_pred, y_prob):
        """
        تحليل الأخطاء والحالات الصعبة
        """
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        print(f"\nError Analysis:")
        print(f"Total errors: {len(error_indices)}")
        print(f"Error rate: {len(error_indices)/len(y_true)*100:.2f}%")

        # أكثر الأخطاء شيوعاً
        error_pairs = list(zip(y_true[errors], y_pred[errors]))
        error_counts = pd.Series(error_pairs).value_counts()

        print(f"\nMost common errors:")
        for (true_label, pred_label), count in error_counts.head(10).items():
            print(f"True: {true_label}, Predicted: {pred_label}, Count: {count}")

        # الحالات الأقل ثقة
        confidence_scores = np.max(y_prob, axis=1)
        low_confidence = np.argsort(confidence_scores)[:20]

        print(f"\nLowest confidence predictions:")
        for idx in low_confidence:
            print(f"Index: {idx}, True: {y_true[idx]}, Pred: {y_pred[idx]}, "
                  f"Confidence: {confidence_scores[idx]:.3f}")

    def plot_class_performance(self, y_true, y_pred):
        """
        رسم أداء كل فئة
        """
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

        x = np.arange(10)
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(range(10))
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('./saved_models', exist_ok=True)
        plt.savefig('./saved_models/class_performance.png', dpi=300, bbox_inches='tight')
        print("📊 Saved per-class performance plot at ./saved_models/class_performance.png")
        plt.close()

    def full_evaluation(self):
        """
        تقييم شامل للنموذج
        """
        print("Starting comprehensive model evaluation...")
        print("=" * 60)

        # تقييم على مجموعة الاختبار
        y_true, y_pred, y_prob = self.evaluate_on_test_set()

        # رسم مصفوفة الالتباس
        confusion_mat = self.plot_confusion_matrix(y_true, y_pred)

        # إنشاء تقرير التصنيف
        metrics = self.generate_classification_report(y_true, y_pred)

        # تحليل الأخطاء
        self.analyze_errors(y_true, y_pred, y_prob)

        # رسم أداء الفئات
        self.plot_class_performance(y_true, y_pred)

        # حفظ النتائج
        results = {
            'confusion_matrix': confusion_mat.tolist(),
            'metrics': metrics,
            'accuracy': np.mean(y_true == y_pred) * 100
        }

        with open('./saved_models/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)

        print(f"\n✅ Evaluation completed! Results saved to './saved_models/evaluation_results.json'")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./saved_models/best_digit_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path, device=args.device)
    results = evaluator.full_evaluation()
