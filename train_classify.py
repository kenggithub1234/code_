import os

# 1. จัดการสภาพแวดล้อมเพื่อปิดข้อความแจ้งเตือนของ TensorFlow ที่ไม่จำเป็น ให้หน้าจอสะอาด
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    print("=== ระบบจำแนกประเภทรูปภาพด้วย Keras (TensorFlow 2.21) ===")
    
    # 2. ตั้งค่าพารามิเตอร์ของชุดข้อมูล
    DATA_DIR = r"C:\ProjectsCode\yolo_dataset_tool\dataset_classify\data" 
    IMG_WIDTH = 128         # ความกว้างของภาพ
    IMG_HEIGHT = 128        # ความสูงของภาพ
    BATCH_SIZE = 4         # จำนวนภาพที่ประมวลผลพร้อมกันในหนึ่งรอบย่อย
    EPOCHS = 200             # จำนวนรอบในการวนลูปคำนวณเทรนโมเดลทั้งหมด
    
    # ตรวจสอบเบื้องต้นว่าโฟลเดอร์ข้อมูลมีอยู่จริงหรือไม่
    if not os.path.exists(DATA_DIR):
        print(f"[Error] ไม่พบโฟลเดอร์ข้อมูลในพาธ: {DATA_DIR}")
        print("กรุณาสร้างโฟลเดอร์และใส่ข้อมูลคลาส (เช่น โฟลเดอร์ OK และ NG) ก่อนเริ่มรัน")
        return

    # 3. โหลดชุดข้อมูลและแบ่งสัดส่วนโดยอัตโนมัติ (Train 80% / Validation 20%)
    print("\n[1/5] กำลังเตรียมและโหลดข้อมูลจากโฟลเดอร์...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE
    )

    # ดึงรายชื่อคลาสจากโครงสร้างโฟลเดอร์ย่อย
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"-> สำเร็จ: พบทั้งหมด {num_classes} คลาส ได้แก่: {class_names}")

    if num_classes < 2:
        print("\n[Warning] ระบบตรวจพบข้อมูลเพียงคลาสเดียว!")
        print("การทำ Image Classification จำเป็นต้องมีโฟลเดอร์ข้อมูลอย่างน้อย 2 คลาสขึ้นไปเพื่อให้ AI เปรียบเทียบ")
        return

    # 4. เพิ่มประสิทธิภาพการดึงข้อมูลเข้าสู่หน่วยความจำของ CPU
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 5. สร้างโครงสร้างโมเดลโครงข่ายประสาทเทียม (CNN Architecture) สไตล์ Keras 3
    print("\n[2/5] กำลังประกอบโครงสร้างโมเดลระบบ...")
    model = models.Sequential([
        # แก้ไขมิติโครงสร้างที่นี่: ส่งค่า (128, 128, 3) ไปโดยตรงเพื่อป้องกันปัญหา Tuple ซ้อน Tuple
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        
        # ปรับสเกลค่าพิกเซลของภาพจากช่วง 0-255 ให้อยู่ในช่วง 0.0 - 1.0 เพื่อความเสถียรในการคำนวณ
        layers.Rescaling(1./255),
        
        # คอนโวลูชันเลเยอร์ชุดที่ 1 ดึงฟีเจอร์เด่นของภาพ
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # คอนโวลูชันเลเยอร์ชุดที่ 2 เพิ่มความละเอียดในการมองเห็นมิติ
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # แปลงข้อมูลภาพแบบมิติสัมพันธ์ให้เป็นเส้นตรง (Vector)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # เลเยอร์ทางออกสุดท้าย (Output Layer) ใช้จำนวนโหนดเท่ากับคลาสจริง และเปิดด้วย softmax เสมอ
        layers.Dense(num_classes, activation='softmax')
    ])

    # 6. คอมไพล์และตั้งค่าฟังก์ชันวัดผลผิดพลาดด้วย Sparse Categorical Crossentropy เพื่อแก้ปัญหาเรื่อง Rank/Dimension
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # แสดงโครงสร้างตารางโมเดลและจำนวนพารามิเตอร์บนหน้าต่างเทอร์มินัล
    model.summary()

    # 7. เริ่มต้นการรันคำนวณและเทรนโมเดล (Model Fitting)
    # ใส่ shuffle=False เพื่อปิดคำเตือนขัดแย้ง เพราะเราสลับข้อมูลในส่วน tf.data ด้านบนไปแล้ว
    print(f"\n[3/5] เริ่มต้นขั้นตอนคำนวณโมเดล (model.fit) ทั้งหมด {EPOCHS} รอบ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        shuffle=False
    )

    # 8. ประเมินความแม่นยำหลังการคำนวณรอบสุดท้ายเสร็จสิ้น
    print("\n[4/5] กำลังประเมินผลความแม่นยำบนชุดข้อมูลสำหรับตรวจสอบ...")
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"-> ผลลัพธ์: ค่า Loss = {loss:.4f} | ค่าความแม่นยำ (Accuracy) = {accuracy * 100:.2f}%")

    # 9. บันทึกไฟล์โมเดลที่เสร็จสมบูรณ์ลงบนฮาร์ดดิสก์
    MODEL_NAME = "yolo_classifier_model.keras"
    print(f"\n[5/5] กำลังบันทึกโมเดลลงสู่ระบบ...")
    model.save(MODEL_NAME)
    print(f"-> สำเร็จ: บันทึกไฟล์โมเดลเรียบร้อยแล้วในชื่อไฟล์ '{MODEL_NAME}'")
    print("\n=== สิ้นสุดการทำงานทุกขั้นตอนอย่างสมบูรณ์ ===")

if __name__ == "__main__":
    main()
