import os
import numpy as np

# 1. จัดการสภาพแวดล้อมเพื่อปิดข้อความแจ้งเตือนของ TensorFlow ที่ไม่จำเป็น
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def predict_new_image():
    # 2. ตั้งค่าพาธของโมเดล รูปภาพทดสอบ และชื่อคลาส (ต้องเรียงตามโฟลเดอร์ตอนเทรน)
    MODEL_PATH = "yolo_classifier_model.keras"
    IMAGE_PATH = r"dataset_classify\OK_J1_TEST\frame_000000_crop_0.jpg" # ปรับเปลี่ยนชื่อไฟล์ภาพตามต้องการ
    CLASS_NAMES = ['NG_J1', 'OK_J1'] 
    
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    print("=== ระบบทดสอบทายผลรูปภาพใหม่ด้วย Keras Model ===")

    # ตรวจสอบว่ามีไฟล์โมเดลอยู่จริงหรือไม่
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] ไม่พบไฟล์โมเดล '{MODEL_PATH}' กรุณารันไฟล์เทรนโมเดลให้เสร็จก่อน")
        return

    # ตรวจสอบว่ามีไฟล์รูปภาพที่ต้องการทดสอบอยู่จริงหรือไม่
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] ไม่พบไฟล์รูปภาพทดสอบที่: {IMAGE_PATH}")
        print("กรุณานำภาพใหม่มาวางในโฟลเดอร์ และตรวจสอบชื่อไฟล์ให้ถูกต้อง")
        return

    # 3. โหลดโมเดลที่บันทึกไว้ขึ้นมาสู่หน่วยความจำ
    print("\nกำลังโหลดโมเดล...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("-> โหลดโมเดลสำเร็จ!")

    # 4. โหลดรูปภาพและปรับแต่งมิติภาพ (Pre-processing) ให้ตรงสเปกตอนเทรน
    print(f"กำลังประมวลผลรูปภาพ: {os.path.basename(IMAGE_PATH)}")
    img = tf.keras.utils.load_img(
        IMAGE_PATH, 
        target_size=(IMG_WIDTH, IMG_HEIGHT)
    )
    
    # แปลงรูปภาพให้เป็น Array ตัวเลข และเพิ่มมิติ Batch (มิติที่ 4) ให้กับภาพเดี่ยว
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # จากมิติ (128, 128, 3) กลายเป็น (1, 128, 128, 3)

    # 5. สั่งโมเดลทำนายผล (Prediction)
    print("\nกำลังประมวลผลทายภาพด้วย AI...")
    predictions = model.predict(img_array, verbose=0)
    
    # ดึงค่าคะแนนความมั่นใจออกมารถในแต่ละคลาสด้วย Softmax คะแนนย่อย
    score = predictions[0]
    predicted_class_idx = np.argmax(score) # ดึงอินเดกซ์ของคลาสที่ได้คะแนนสูงสุด
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    confidence_percentage = 100 * score[predicted_class_idx]

    # 6. แสดงผลลัพธ์บนหน้าจอ
    print("\n================ ผลลัพธ์การตรวจสอบ ================")
    print(f"-> ภาพนี้ถูกจำแนกเป็นประเภท: [ {predicted_class_name} ]")
    print(f"-> ความมั่นใจของระบบ (Confidence): {confidence_percentage:.2f}%")
    print("==================================================")
    
    # พิมพ์รายละเอียดคะแนนของทุกคลาสเพื่อการวิเคราะห์
    print("\nรายละเอียดคะแนนรายคลาส:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"   - คลาส {name}: ความมั่นใจ {100 * score[idx]:.2f}%")

if __name__ == "__main__":
    predict_new_image()
