import os
import sys
import io
import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class _QtLogStream(io.TextIOBase):
    """Writable stream that forwards each printed line to a Qt signal.
    Used to capture Keras' console output (progress bars use \\r a lot)
    into the on-screen log widget."""

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn
        self._buffer = ""

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer or "\r" in self._buffer:
            idx_n = self._buffer.find("\n")
            idx_r = self._buffer.find("\r")
            candidates = [i for i in (idx_n, idx_r) if i != -1]
            idx = min(candidates)
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]
            if line.strip():
                self._emit_fn(line)
        return len(s)

    def flush(self):
        pass


class KerasTrainWorker(QThread):
    log_signal = pyqtSignal(str)
    epoch_progress_signal = pyqtSignal(int, int)  # current_epoch, total_epochs
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, dataset_dir, architecture, img_size, batch_size, epochs,
                 learning_rate, augmentation, output_dir, run_name):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.architecture = architecture
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.augmentation = augmentation  # dict of toggle flags + values
        self.output_dir = output_dir
        self.run_name = run_name

    def run(self):
        # Suppress noisy TensorFlow startup/oneDNN log spam. This MUST be
        # set before tensorflow is ever imported in this process to take
        # effect (TF reads it once at C++ core init time).
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        old_stdout, old_stderr = sys.stdout, sys.stderr
        stream = _QtLogStream(self.log_signal.emit)
        sys.stdout = stream
        sys.stderr = stream
        try:
            self._run_impl()
        except Exception as e:
            self.log_signal.emit("เกิดข้อผิดพลาด: " + str(e))
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _run_impl(self):
        try:
            import tensorflow as tf
        except ImportError:
            self.finished_signal.emit(
                False,
                "ไม่พบไลบรารี tensorflow กรุณาติดตั้งก่อนด้วยคำสั่ง: pip install tensorflow",
            )
            return

        # NOTE: tf.keras.preprocessing.image.ImageDataGenerator is deprecated
        # (since TF 2.9) and has been removed entirely under Keras 3, which
        # is the default from TF 2.16 onward. We use the current recommended
        # approach instead: tf.keras.utils.image_dataset_from_directory for
        # loading, plus Keras preprocessing/augmentation *layers* baked
        # directly into the model. Those augmentation layers automatically
        # no-op during validation/inference (Keras handles this internally),
        # and the saved model.keras ends up fully self-contained - no
        # separate preprocessing step needs to be remembered when reusing it.

        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")
        img_size = self.img_size

        self.log_signal.emit("กำลังเตรียมข้อมูล...")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir, image_size=(img_size, img_size), batch_size=self.batch_size,
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir, image_size=(img_size, img_size), batch_size=self.batch_size, shuffle=False,
            )
        else:
            self.log_signal.emit(
                "ไม่พบโฟลเดอร์ train/ และ val/ แยกกัน - จะแบ่ง validation 20% จากโฟลเดอร์เดียวกันแทน "
                "(แนะนำให้ใช้ปุ่ม 'แบ่ง Train/Val อัตโนมัติ' ในแท็บ Classify ก่อน เพื่อผลลัพธ์ที่แม่นยำกว่า)"
            )
            train_ds = tf.keras.utils.image_dataset_from_directory(
                self.dataset_dir, validation_split=0.2, subset="training", seed=123,
                image_size=(img_size, img_size), batch_size=self.batch_size,
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.dataset_dir, validation_split=0.2, subset="validation", seed=123,
                image_size=(img_size, img_size), batch_size=self.batch_size,
            )

        class_names = train_ds.class_names
        num_classes = len(class_names)
        if num_classes < 2:
            self.finished_signal.emit(
                False, "ต้องมีอย่างน้อย 2 คลาสใน dataset จึงจะเทรน classification ได้"
            )
            return

        self.log_signal.emit(f"พบ {num_classes} คลาส: {class_names}")

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        self.log_signal.emit(f"กำลังสร้างโมเดล ({self.architecture})...")
        model = self._build_model(tf, num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            # Labels come out of image_dataset_from_directory as plain
            # integers by default (not one-hot), so sparse_categorical_
            # crossentropy is the matching loss - avoids rank/shape
            # mismatch errors some Keras/TF version combos raise with
            # one-hot "categorical_crossentropy" here.
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary(print_fn=lambda line: self.log_signal.emit(line))

        run_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=True)

        worker_self = self

        class _ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                worker_self.epoch_progress_signal.emit(epoch + 1, worker_self.epochs)
                worker_self.log_signal.emit(
                    f"Epoch {epoch + 1}/{worker_self.epochs} - "
                    f"loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - "
                    f"val_loss: {logs.get('val_loss', 0):.4f} - val_acc: {logs.get('val_accuracy', 0):.4f}"
                )

        self.log_signal.emit("เริ่มการเทรน...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=[_ProgressCallback()],
            shuffle=False,  # already shuffled via .shuffle(1000) on the tf.data pipeline above
            verbose=1,
        )

        self.log_signal.emit("กำลังประเมินผลความแม่นยำบนชุดข้อมูลสำหรับตรวจสอบ...")
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        self.log_signal.emit(
            f"ผลลัพธ์สุดท้าย: Loss = {val_loss:.4f} | Accuracy = {val_accuracy * 100:.2f}%"
        )

        model_path = os.path.join(run_dir, "model.keras")
        model.save(model_path)

        with open(os.path.join(run_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(class_names))
        # Preprocessing/rescaling is now baked into the model itself (see
        # _build_model), so anything reusing this model - including this
        # app's own Auto-label feature in the Classify tab - should feed it
        # raw, unscaled 0-255 pixels and pick "raw" there.
        with open(os.path.join(run_dir, "recommended_preprocessing.txt"), "w", encoding="utf-8") as f:
            f.write("raw")

        try:
            self._save_history_plot(history, run_dir)
        except Exception as e:
            self.log_signal.emit(f"(ไม่สามารถสร้างกราฟ history ได้: {e})")

        self.finished_signal.emit(True, model_path)

    def _build_augmentation_layers(self, tf):
        aug = self.augmentation
        layers = []

        if aug.get("use_rotation"):
            # RandomRotation's factor is a fraction of a full turn (2*pi),
            # while the UI collects a max angle in degrees.
            rotation_factor = aug.get("rotation_range", 20) / 360.0
            layers.append(tf.keras.layers.RandomRotation(rotation_factor))

        if aug.get("use_width_shift") or aug.get("use_height_shift"):
            width_factor = aug.get("width_shift_range", 0.0) if aug.get("use_width_shift") else 0.0
            height_factor = aug.get("height_shift_range", 0.0) if aug.get("use_height_shift") else 0.0
            layers.append(tf.keras.layers.RandomTranslation(
                height_factor=height_factor, width_factor=width_factor
            ))

        if aug.get("use_brightness"):
            b_min = aug.get("brightness_min", 0.7)
            b_max = aug.get("brightness_max", 1.3)
            # Convert the old ImageDataGenerator-style multiplicative range
            # (values around 1.0) into RandomBrightness's symmetric additive
            # factor (0-1, fraction of the value_range to jitter by).
            delta = min(max(abs(1.0 - b_min), abs(b_max - 1.0)), 1.0)
            layers.append(tf.keras.layers.RandomBrightness(factor=delta, value_range=(0, 255)))

        return layers

    def _build_model(self, tf, num_classes):
        img_size = self.img_size
        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        x = inputs

        aug_layers = self._build_augmentation_layers(tf)
        if aug_layers:
            # Wrapped in its own Sequential purely for a tidy name in the
            # model summary; Keras automatically disables every layer here
            # (Random*) during validation/inference.
            x = tf.keras.Sequential(aug_layers, name="augmentation")(x)

        if self.architecture == "mobilenetv2":
            # Mathematically identical to tf.keras.applications.mobilenet_v2
            # .preprocess_input (scales 0-255 -> -1..1), but expressed as a
            # plain layer so it serializes cleanly and needs no external
            # preprocessing step when the saved model is reused later.
            x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
            base = tf.keras.applications.MobileNetV2(
                include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
            )
            base.trainable = False
            x = base(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        elif self.architecture == "efficientnetb0":
            # EfficientNet's Keras implementation has its own internal
            # Normalization layer and expects raw 0-255 pixels, so no extra
            # rescaling layer is added here.
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
            )
            base.trainable = False
            x = base(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        else:  # simple_cnn trained from scratch
            x = tf.keras.layers.Rescaling(scale=1.0 / 255)(x)
            x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs)

    def _save_history_plot(self, history, run_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.history.get("accuracy", []), label="train")
        axes[0].plot(history.history.get("val_accuracy", []), label="val")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history.history.get("loss", []), label="train")
        axes[1].plot(history.history.get("val_loss", []), label="val")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "training_history.png"))
        plt.close(fig)
