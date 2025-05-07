#!/usr/bin/env python
"""
train_se_aug_wd_ls.py

Final model: ResNet-50 + SE + flip augmentation + weight decay + label smoothing.
Trains for up to 50 epochs, with LR reduction on plateau and early stopping,
plots loss/accuracy curves, reports final test accuracy, and plots a
row-normalized confusion matrix with emotion labels.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Reshape, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# -------------------------------
# Config
# -------------------------------
IMG_SIZE     = 224
NUM_CLASSES  = 8
CSV_PATH     = 'affectnet_bbox.csv'
BATCH_SIZE   = 64
EPOCHS       = 50
MARGIN       = 0.15
BASE_LR      = 1e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTH = 0.1

# Emotion labels for the confusion matrix
EMOTIONS = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

# -------------------------------
# Squeeze-and-Excitation block
# -------------------------------
def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = Dense(filters//reduction, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = Reshape((1,1,filters))(se)
    return Multiply()([x, se])

# -------------------------------
# parsing + preprocessing
# -------------------------------
def parse_fn(path, xmin, ymin, xmax, ymax, label):
    xmin = tf.cast(xmin, tf.int32)
    ymin = tf.cast(ymin, tf.int32)
    xmax = tf.cast(xmax, tf.int32)
    ymax = tf.cast(ymax, tf.int32)

    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    w, h = xmax - xmin, ymax - ymin
    dx = tf.cast(tf.cast(w, tf.float32) * MARGIN, tf.int32)
    dy = tf.cast(tf.cast(h, tf.float32) * MARGIN, tf.int32)

    x1 = tf.maximum(0, xmin - dx)
    y1 = tf.maximum(0, ymin - dy)
    x2 = tf.minimum(tf.shape(img)[1], xmax + dx)
    y2 = tf.minimum(tf.shape(img)[0], ymax + dy)
    face = img[y1:y2, x1:x2, :]

    face = tf.cond(
        tf.logical_or(tf.equal(tf.shape(face)[0],0),
                      tf.equal(tf.shape(face)[1],0)),
        lambda: img,
        lambda: face
    )

    face = tf.image.resize(face, [IMG_SIZE, IMG_SIZE])
    face = tf.keras.applications.resnet50.preprocess_input(face)
    lbl  = tf.one_hot(label, NUM_CLASSES)
    return face, lbl

# -------------------------------
# Dataset builder
# -------------------------------
def df_to_dataset(df, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['x_min'].values.astype(np.int32),
        df['y_min'].values.astype(np.int32),
        df['x_max'].values.astype(np.int32),
        df['y_max'].values.astype(np.int32),
        df['expression'].values.astype(np.int32),
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(
            lambda img, lbl: (tf.image.random_flip_left_right(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Load & split
# -------------------------------
df = pd.read_csv(CSV_PATH)
train_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df['expression'], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1765,
    stratify=train_df['expression'], random_state=42
)

ds_train = df_to_dataset(train_df, shuffle=True,  augment=True)
ds_val   = df_to_dataset(val_df,   shuffle=False, augment=False)
ds_test  = df_to_dataset(test_df,  shuffle=False, augment=False)
y_true   = test_df['expression'].values.astype(int)

# -------------------------------
# Build model
# -------------------------------
def build_se_aug_wd_ls():
    inp  = Input((IMG_SIZE, IMG_SIZE, 3))
    base = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_tensor=inp
    )
    x    = se_block(base.output)
    x    = GlobalAveragePooling2D()(x)
    out  = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inp, out, name='se_aug_wd_ls')
    model.compile(
        optimizer=AdamW(learning_rate=BASE_LR, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy']
    )
    return model

# -------------------------------
# Callbacks
# -------------------------------
checkpoint = ModelCheckpoint(
    'se_aug_wd_ls_best.h5',
    monitor='val_accuracy', save_best_only=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5,
    min_lr=1e-6, verbose=1
)
earlystop = EarlyStopping(
    monitor='val_loss', patience=7,
    restore_best_weights=True, verbose=1
)

# -------------------------------
# Train
# -------------------------------
model = build_se_aug_wd_ls()
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, earlystop],
    verbose=1
)

# -------------------------------
# Plot Loss & Accuracy
# -------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],      label='Train')
plt.plot(history.history['val_loss'],  label='Val')
plt.title('SE+Aug+WD+LS Loss')
plt.xlabel('Epoch'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],      label='Train')
plt.plot(history.history['val_accuracy'],  label='Val')
plt.title('SE+Aug+WD+LS Accuracy')
plt.xlabel('Epoch'); plt.legend()

plt.tight_layout()
plt.savefig('se_aug_wd_ls_loss_acc.png')
plt.show()

# -------------------------------
# Final Evaluation
# -------------------------------
test_loss, test_acc = model.evaluate(ds_test, verbose=1)
print(f'\nFinal test accuracy: {test_acc:.4f}')

# -------------------------------
# Confusion Matrix
# -------------------------------
y_pred      = model.predict(ds_test, verbose=1)
pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, pred_labels, normalize='true')
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig('se_aug_wd_ls_confusion_matrix.png')
plt.show()

# -------------------------------
# Save Model
# -------------------------------
model.save('se_aug_wd_ls_final.h5')
print('Saved se_aug_wd_ls_final.h5')
