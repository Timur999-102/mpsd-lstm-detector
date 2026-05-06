import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task, Dataset
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, f1_score

# === ClearML Task ===
task = Task.init(
    project_name="PowerShell_Malware_Detection",
    task_name="LSTM_Remote_Training",
    task_type=Task.TaskTypes.training,
    output_uri=True
)

# 🔥 Инициализируем логгер (это было пропущено)
logger = task.get_logger()

# Гиперпараметры
task.connect_configuration({
    "max_tokens_vocab": 90000,
    "embedding_dim": 32,
    "lstm_units": [64, 32],
    "dropout_rate": 0.3,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 1e-4,
    "sequence_length_percentile": 95
})

# 🔥 Отправка задачи в очередь Colab.
# Весь код НИЖЕ этой строки будет выполняться УЖЕ НА АГЕНТЕ в Colab.
task.execute_remotely(queue_name="colab_gpu")

print("🚀 Скрипт запущен на удаленном агенте!")

# === Загрузка датасета ===
print("📥 Загрузка датасета из ClearML...")

dataset = Dataset.get(
    dataset_project="PowerShell_Malware",
    dataset_name="MPSD_Dataset_v2"
)
data_path = dataset.get_local_copy()  # Автоматически скачает/возьмёт из кэша
print(f"✅ Датасет загружен: {data_path}")

# === Функции предобработки ===
def read_files(path):
    dataset = []
    if not os.path.exists(path): return dataset
    for f in os.listdir(path):
        if not f.endswith('.ps1'): continue
        try:
            with open(os.path.join(path, f), encoding='utf-8', errors='ignore') as file:
                content = file.read().replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
                dataset.append(re.sub(r'\s+', ' ', content.strip()))
        except:
            pass
    return dataset

def clean_data_to_tokens(dataset):
    tokens = []
    char_replace = "()[]{},;'/\\=:^<>|`+\"&$"
    for s in dataset:
        s = re.sub(r"FromBase64String\s*\(\s*['\"]?([A-Za-z0-9+/=]{50,})['\"]?\s*\)", " [BASE64_PAYLOAD] ", s,
                   flags=re.IGNORECASE)
        s = re.sub(r"([A-Za-z0-9+/=]{100,})", " [BASE64_PAYLOAD] ", s)
        ip_pattern = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        for ip in re.findall(ip_pattern, s):
            s = s.replace(ip, ' [INTERNAL_IP] ' if ip.startswith(
                ('10.', '192.168.', '172.16.', '127.')) else ' [EXTERNAL_IP] ')
        s = re.sub(r"\b0x[A-Fa-f0-9]+\b", " [HEX_VALUE] ", s)
        s = re.sub(r"\b\d{4,}\b", " [LONG_NUMBER] ", s)
        for c in char_replace: s = s.replace(c, ' ')
        tokens.append(re.sub(r'\s+', ' ', s.lower()).strip())
    return tokens

def get_optimal_sequence_length(tokenized_data, percentile=97):
    lengths = [len(s.split()) for s in tokenized_data]
    return int(np.percentile(lengths, percentile)) if lengths else 512

# === Загрузка данных ===
train_bad_pure = read_files(os.path.join(data_path, 'malicious_pure'))
train_bad_mixed = read_files(os.path.join(data_path, 'mixed_malicious'))
train_valid = read_files(os.path.join(data_path, 'powershell_benign_dataset'))
test_bad = read_files(os.path.join(data_path, 'test_ds_bad'))
test_valid = read_files(os.path.join(data_path, 'test_ds_valid'))

train_bad = train_bad_pure + train_bad_mixed
train_ds = train_bad + train_valid
test_ds = test_bad + test_valid

# Метки: 1 = вредоносный, 0 = чистый
y_train = [1] * len(train_bad) + [0] * len(train_valid)
y_test = [1] * len(test_bad) + [0] * len(test_valid)
print(f"""
✅ Загружено:
├─ malicious_pure: {len(train_bad_pure)}
├─ mixed_malicious: {len(train_bad_mixed)}
├─ benign: {len(train_valid)}
├─ Train total: {len(train_ds)} (malicious={len(train_bad)}, benign={len(train_valid)})
└─ Test: {len(test_ds)}
""")

# === Токенизация ===
train_token_ds = clean_data_to_tokens(train_ds)
test_token_ds = clean_data_to_tokens(test_ds)
max_token = get_optimal_sequence_length(train_token_ds, percentile=95)
print(f"📏 Длина последовательности: {max_token}")

# === Разделение ===
train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    train_token_ds, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# === Векторизация ===
max_tokens_count = 90000
text_vectorizer = layers.TextVectorization(
    max_tokens=max_tokens_count,
    output_sequence_length=max_token,
    standardize=None
)
text_vectorizer.adapt(train_tokens)

# 🔥 ИСПРАВЛЕНИЕ БАГА KERAS 3
vocab = text_vectorizer.get_vocabulary()
vocab_clean = [w for w in vocab if w not in ('', '[UNK]')]
text_vectorizer.set_vocabulary(['', '[UNK]'] + vocab_clean)

# === Модель ===
def build_lstm_model(vocab_size, embedding_dim, lstm_units, dropout_rate):
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units[1], dropout=dropout_rate))(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    return models.Model(inputs, outputs)

model_pwsh = build_lstm_model(
    vocab_size=max_tokens_count,
    embedding_dim=32,
    lstm_units=[64, 32],
    dropout_rate=0.3
)

# === Класс для логирования по батчам ===
class ClearMLBatchLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger, log_every_n_batches=50):
        self.logger = logger
        self.global_step = 0
        self.log_every_n_batches = log_every_n_batches

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        logs = logs or {}

        if self.global_step % self.log_every_n_batches == 0:
            self.logger.report_scalar(title="Loss", series="Train", value=logs.get("loss", 0),
                                      iteration=self.global_step)
            self.logger.report_scalar(title="Accuracy", series="Train", value=logs.get("accuracy", 0),
                                      iteration=self.global_step)

            p = logs.get("precision", 0)
            r = logs.get("recall", 0)
            f1 = 2 * (p * r) / (p + r + 1e-7)

            self.logger.report_scalar(title="Precision", series="Train", value=p, iteration=self.global_step)
            self.logger.report_scalar(title="Recall", series="Train", value=r, iteration=self.global_step)
            self.logger.report_scalar(title="F1-Score", series="Train", value=f1, iteration=self.global_step)
            self.logger.report_scalar(title="ROC-AUC", series="Train", value=logs.get("auc", 0),
                                      iteration=self.global_step)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.logger.report_scalar(title="Loss", series="Validation", value=logs.get("val_loss", 0),
                                  iteration=self.global_step)
        self.logger.report_scalar(title="Accuracy", series="Validation", value=logs.get("val_accuracy", 0),
                                  iteration=self.global_step)

        val_p = logs.get("val_precision", 0)
        val_r = logs.get("val_recall", 0)
        val_f1 = 2 * (val_p * val_r) / (val_p + val_r + 1e-7)

        self.logger.report_scalar(title="Precision", series="Validation", value=val_p, iteration=self.global_step)
        self.logger.report_scalar(title="Recall", series="Validation", value=val_r, iteration=self.global_step)
        self.logger.report_scalar(title="F1-Score", series="Validation", value=val_f1, iteration=self.global_step)
        self.logger.report_scalar(title="ROC-AUC", series="Validation", value=logs.get("val_auc", 0),
                                  iteration=self.global_step)

model_pwsh.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# === Callbacks ===
callbacks_list = [
    # Убрали жесткий путь /content/, сохраняем в рабочей директории агента
    callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max',
                              verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
    ClearMLBatchLogger(logger, log_every_n_batches=50)
]

# === Подготовка данных ===
encoder = OneHotEncoder(sparse_output=False)
y_train_enc = encoder.fit_transform(np.array(train_labels).reshape(-1, 1))
y_val_enc = encoder.transform(np.array(val_labels).reshape(-1, 1))

train_dataset = tf.data.Dataset.from_tensor_slices((train_tokens, y_train_enc)).shuffle(1000).batch(16).prefetch(
    tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_tokens, y_val_enc)).batch(16).prefetch(tf.data.AUTOTUNE)

# === Обучение ===
print("\n🚀 Обучение...")
history = model_pwsh.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=callbacks_list,
    verbose=1
)

logger.report_table(
    title="Dataset Composition",
    series="Training Data",
    iteration=0,
    table_plot=pd.DataFrame({
        "Type": ["malicious_pure", "mixed_malicious", "benign"],
        "Count": [len(train_bad_pure), len(train_bad_mixed), len(train_valid)],
        "Label": [1, 1, 0]
    })
)

# === Визуализация результатов ===
val_probs = model_pwsh.predict(val_dataset, verbose=0)
val_preds = np.argmax(val_probs, axis=1)
val_labels_decoded = np.argmax(y_val_enc, axis=1)

acc = accuracy_score(val_labels_decoded, val_preds)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels_decoded, val_preds, average='weighted', zero_division=0)
roc_auc = roc_auc_score(val_labels_decoded, val_probs[:, 1]) if len(np.unique(val_labels_decoded)) > 1 else 0.5

print(f"""
📈 Результаты:
├─ Accuracy:  {acc:.4f}
├─ Precision: {precision:.4f}
├─ Recall:    {recall:.4f}
├─ F1-Score:  {f1:.4f}
└─ ROC-AUC:   {roc_auc:.4f}
""")

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy по эпохам')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
logger.report_matplotlib_figure(title="Training Metrics", series="Accuracy", figure=plt)
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss по эпохам')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
logger.report_matplotlib_figure(title="Training Metrics", series="Loss", figure=plt)
plt.close()

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']
values = [f1, precision, recall, roc_auc]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

plt.figure(figsize=(8, 5))
bars = plt.barh(metrics, values, color=colors)
plt.xlim(0, 1.1)
plt.xlabel('Значение')
plt.title('Сводные метрики модели')
for bar, val in zip(bars, values):
    plt.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=9)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
logger.report_matplotlib_figure(title="Model Metrics", series="Summary", figure=plt)
plt.close()

cm = confusion_matrix(val_labels_decoded, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix (Validation)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
logger.report_confusion_matrix(title="Model Metrics", series="Confusion Matrix", matrix=cm, xaxis="Predicted", yaxis="Actual")
plt.close()

if len(np.unique(val_labels_decoded)) > 1:
    fpr, tpr, _ = roc_curve(val_labels_decoded, val_probs[:, 1])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    logger.report_matplotlib_figure(title="Model Metrics", series="ROC Curve", figure=plt)
    plt.close()

print("✅ Все графики отправлены в ClearML!")

# 🔥 Оценка на mixed-данных
if len(test_bad) > 0 and any('mixed' in str(path).lower() for path in
    [os.path.join(data_path, 'test_ds_bad', f) for f in
        os.listdir(os.path.join(data_path, 'test_ds_bad')) if f.endswith('.ps1')]):

    test_probs = model_pwsh.predict(
        tf.data.Dataset.from_tensor_slices(test_token_ds[:len(test_bad)]).batch(16),
        verbose=0
    )
    test_preds = np.argmax(test_probs, axis=1)

    mixed_accuracy = accuracy_score(y_test[:len(test_bad)], test_preds)
    mixed_f1 = f1_score(y_test[:len(test_bad)], test_preds, average='weighted')

    print(f"""
🎯 Результаты на вредоносных (включая mixed):
├─ Accuracy: {mixed_accuracy:.4f}
└─ F1-Score: {mixed_f1:.4f}
""")

    logger.report_scalar(title="Test Metrics", series="mixed_accuracy", value=mixed_accuracy, iteration=0)
    logger.report_scalar(title="Test Metrics", series="mixed_f1", value=mixed_f1, iteration=0)

# === Сохранение и завершение ===
import json

# Сохраняем в относительные пути
model_path = 'final_lstm_model.keras'
metrics_path = 'metrics.json'

model_pwsh.save(model_path)

results = {
    'accuracy': float(acc), 'precision': float(precision),
    'recall': float(recall), 'f1': float(f1), 'roc_auc': float(roc_auc)
}
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=2)

try:
    task.upload_artifact('model', model_path)
    task.upload_artifact('metrics', metrics_path)
    print("📤 Артефакты загружены в ClearML")
except Exception as e:
    print(f"⚠️ Ошибка при загрузке артефактов: {e}")

task.close()
print(f"✅ Обучение завершено! Ссылка: {task.get_output_log_web_page()}")