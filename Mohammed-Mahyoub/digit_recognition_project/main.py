import argparse
import sys
import os

# إضافة مجلد src للمسار
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='نظام التعرف على الأرقام المحسن')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict', 'gui'], default='gui', help='وضع التشغيل')
    parser.add_argument('--model-path', default='./saved_models/best_digit_model.pth', help='مسار النموذج المحفوظ')
    parser.add_argument('--image-path', help='مسار الصورة للتنبؤ')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        from src.train_emnist import DigitTrainer
        print("🚀 بدء تدريب النموذج...")
        trainer = DigitTrainer(model_type='improved', learning_rate=0.001)
        trainer.train(num_epochs=25)
        
    elif args.mode == 'evaluate':
        from src.evaluate import ModelEvaluator
        print("📊 بدء تقييم النموذج...")
        if not os.path.exists(args.model_path):
            print(f"❌ النموذج غير موجود في {args.model_path}")
            print("💡 قم بتدريب النموذج أولاً باستخدام: python main.py --mode train")
            return
        
        evaluator = ModelEvaluator(args.model_path)
        evaluator.full_evaluation()
        
    elif args.mode == 'predict':
        from src.predict import ImprovedDigitPredictor
        if not args.image_path:
            print("❌ يجب تحديد مسار الصورة للتنبؤ")
            print("💡 استخدم: python main.py --mode predict --image-path path/to/image.jpg")
            return
        
        if not os.path.exists(args.image_path):
            print(f"❌ الصورة غير موجودة في {args.image_path}")
            return
        
        print("🔍 بدء التحليل...")
        predictor = ImprovedDigitPredictor(args.model_path)
        predictor.visualize_prediction(args.image_path)
        
    elif args.mode == 'gui':
        # import gui
        from src.gui import DigitRecognitionGUI
        print("🚀 تشغيل الواجهة الرسومية...")
        app = DigitRecognitionGUI(args.model_path)
        app.run()

if __name__ == "__main__":
    print("=" * 60)
    print("🔢 نظام التعرف على الأرقام المحسن")
    print("Enhanced Digit Recognition System")
    print("=" * 60)
    
    main()
