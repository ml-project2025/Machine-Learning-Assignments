import gradio as gr
import numpy as np
import cv2
import tensorflow as tf


model = tf.keras.models.load_model("models/digit_cnn.keras")

# هنا الخاص بالصور 
def process_upload(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_inverted = cv2.bitwise_not(image_gray)
    image_resized = cv2.resize(image_inverted, (28, 28))
    preview = image_resized.copy()
    image_normalized = image_resized.astype("float32") / 255.0
    final_image = np.expand_dims(image_normalized, axis=-1)
    final_image = np.expand_dims(final_image, axis=0)
    return final_image, preview
# وهنا الخاص بحق الرسم 
def process_sketch(sketch_data):
    if not isinstance(sketch_data, dict):
        raise ValueError("المدخل من Sketchpad ليس قاموسًا كما هو متوقع.")
    image_numpy = None
    if 'image' in sketch_data and sketch_data['image'] is not None:
        image_numpy = sketch_data['image']
    elif 'composite' in sketch_data and sketch_data['composite'] is not None:
        image_numpy = sketch_data['composite']
    if image_numpy is None:
        raise ValueError("صورة الرسم فارغة أو غير صالحة.")
    if image_numpy.shape[2] == 4:
        image_gray = cv2.cvtColor(image_numpy, cv2.COLOR_RGBA2GRAY)
    elif image_numpy.shape[2] == 3:
        image_gray = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_numpy[:, :, 0] if image_numpy.ndim == 3 else image_numpy
    image_resized = cv2.resize(image_gray, (28, 28))
    image_inverted = cv2.bitwise_not(image_resized)
    processed_image = cv2.adaptiveThreshold(image_inverted, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    preview = processed_image.copy()
    image_normalized = processed_image.astype("float32") / 255.0
    final_image = np.expand_dims(image_normalized, axis=-1)
    final_image = np.expand_dims(final_image, axis=0)
    return final_image, preview

def predict_digit(processed_image):
    pred = model.predict(processed_image, verbose=0)
    confidences = {str(i): float(pred[0][i]) for i in range(10)}
    predicted_digit = int(np.argmax(pred))
    return predicted_digit, confidences

custom_css = """
footer, footer a, footer div {
    display: none !important;
}
"""
# الواجهة
with gr.Blocks(theme=gr.themes.Soft(),title="نظام التعرف على الأرقام",css=custom_css) as demo:
    gr.Markdown("# تطبيق التعرف على الأرقام")
    with gr.Tabs():
        with gr.TabItem("📷 رفع صورة"):
            with gr.Row():
                upload_input = gr.Image(type="numpy", label="ارفع صورة رقم")
                with gr.Column():
                    upload_output_digit = gr.Number(label="التوقع")
                    upload_output_conf = gr.Label(label="الاحتمالات")
                    upload_preview = gr.Image(label="المعاينة")
            upload_button = gr.Button("تعرف على الصورة")
        with gr.TabItem("✏️ ارسم رقمًا"):
            with gr.Row():
                sketch_input = gr.Sketchpad(label="ارسم هنا")
                with gr.Column():
                    sketch_output_digit = gr.Number(label="التوقع")
                    sketch_output_conf = gr.Label(label="الاحتمالات")
                    sketch_preview = gr.Image(label="المعاينة")
            sketch_button = gr.Button("تعرف على الرسمة")

    def handle_upload(img):
        if img is None:
            return None, None, None
        processed_img, preview_img = process_upload(img)
        digit, confidences = predict_digit(processed_img)
        return digit, confidences, preview_img

    def handle_sketch(sketch_data):
        if sketch_data is None:
            return None, None, None
        try:
            processed_img, preview_img = process_sketch(sketch_data)
            digit, confidences = predict_digit(processed_img)
            return digit, confidences, preview_img
        except Exception as e:
            return "خطأ: " + str(e), None, None

    upload_button.click(fn=handle_upload, inputs=upload_input,
                         outputs=[upload_output_digit, upload_output_conf, upload_preview])
    sketch_button.click(fn=handle_sketch, inputs=sketch_input,
                         outputs=[sketch_output_digit, sketch_output_conf, sketch_preview])

demo.launch()
