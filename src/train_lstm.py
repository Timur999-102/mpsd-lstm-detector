import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from clearml import Task, Dataset

# === Диагностика импортов (добавь в начало src/train_lstm.py) ===
print(f"🐍 Python: {sys.version}")
print(f"📁 CWD: {os.getcwd()}")
print(f"📄 Executing: {__file__}")

try:
    from clearml import Task, Dataset
    print("✅ ClearML imports: OK")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
# === Конец диагностики ===

# === Оптимизация для стабильности ===
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Подавить лишние предупреждения TF

# 🔐 ClearML API-ключи
os.environ['CLEARML_API_ACCESS_KEY'] = 'S78S76DYLI0C6AM97ASTOHIEECNNX2'
os.environ['CLEARML_API_SECRET_KEY'] = 'i52pJEwFecfPR2dlK9DY53V0IbfG7jUgiEYeSycmtSx80wJlfXIGPpuEpuHoBikMDdY'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'

# === Настройка путей ===
# Корень проекта = папка, в которой лежит src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
print(f"📁 MODELS_PATH: {MODELS_PATH}")


#Переопределяем точку входа
os.environ['CLEARML_SCRIPT_ENTRY_POINT'] = 'src/train_lstm.py'

# === ClearML Task ===
task = Task.init(
    project_name="PowerShell_Malware_Detection",
    task_name="LSTM_Local_Training",  # 🔥 Имя для локального запуска
    task_type=Task.TaskTypes.training,
    output_uri=MODELS_PATH,  # 🔥 Сохранять артефакты в папку models/
    reuse_last_task_id=False,

    # 🔥 Для консистентности (опционально):
    #script_repo="https://github.com/Timur999-102/mpsd-lstm-detector.git",
    #script_path="src/train_lstm.py"
)

# Гиперпараметры
task.connect_configuration({
    "max_tokens_vocab": 90000,
    "embedding_dim": 32,
    "lstm_units": [64, 32],
    "dropout_rate": 0.3,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 1e-4,
    "sequence_length_percentile": 95
})

# 🔥 Для локального запуска — закомментировано!
#task.execute_remotely(queue_name="default")

logger = task.get_logger()

# === Загрузка датасета ===
# Вариант А: Из ClearML Dataset (работает и локально, и на агенте)
print("📥 Загрузка датасета из ClearML...")
try:
    dataset = Dataset.get(
        dataset_project="PowerShell_Malware",
        dataset_name="MPSD_Dataset_v2"
    )
    data_path = dataset.get_local_copy()  # Автоматически скачает/возьмёт из кэша
    print(f"✅ Датасет загружен: {data_path}")
except Exception as e:
    print(f"⚠️ Не удалось загрузить из ClearML: {e}")
    print("💡 Пробуем локальную папку...")
    data_path = os.path.join(PROJECT_ROOT, 'data', 'raw')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Датасет не найден ни в ClearML, ни локально: {data_path}")

# === Функции предобработки ===
def read_files(path):
    dataset = []
    if not os.path.exists(path):
        print(f"⚠️ Путь не найден: {path}")
        return dataset
    for f in os.listdir(path):
        if not f.endswith('.ps1'):
            continue
        try:
            with open(os.path.join(path, f), encoding='utf-8', errors='ignore') as file:
                content = file.read().replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
                dataset.append(re.sub(r'\s+', ' ', content.strip()))
        except Exception as ex:
            print(f"⚠️ Ошибка чтения {f}: {ex}")
    return dataset

def clean_data_to_tokens(dataset):
    tokens = []
    char_replace = "()[]{},;'/\\=:^<>|`+\"&$"
    for s in dataset:
        s = re.sub(r"FromBase64String\s*\(\s*['\"]?([A-Za-z0-9+/=]{50,})['\"]?\s*\)", " [BASE64_PAYLOAD] ", s, flags=re.IGNORECASE)
        s = re.sub(r"([A-Za-z0-9+/=]{100,})", " [BASE64_PAYLOAD] ", s)
        ip_pattern = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        for ip in re.findall(ip_pattern, s):
            s = s.replace(ip, ' [INTERNAL_IP] ' if ip.startswith(('10.','192.168.','172.16.','127.')) else ' [EXTERNAL_IP] ')
        s = re.sub(r"\b0x[A-Fa-f0-9]+\b", " [HEX_VALUE] ", s)
        s = re.sub(r"\b\d{4,}\b", " [LONG_NUMBER] ", s)
        for c in char_replace:
            s = s.replace(c, ' ')
        tokens.append(re.sub(r'\s+', ' ', s.lower()).strip())
    return tokens

def get_optimal_sequence_length(tokenized_data, percentile=95):
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

y_train = [1]*len(train_bad) + [0]*len(train_valid)
y_test = [1]*len(test_bad) + [0]*len(test_valid)

print(f"""
✅ Загружено:
├─ malicious_pure: {len(train_bad_pure)}
├─ mixed_malicious: {len(train_bad_mixed)}
├─ benign: {len(train_valid)}
├─ Train: {len(train_ds)} (malicious={len(train_bad)}, benign={len(train_valid)})
└─ Test: {len(test_ds)}
""")

# === Токенизация ===
train_token_ds = clean_data_to_tokens(train_ds)
test_token_ds = clean_data_to_tokens(test_ds)
max_token = get_optimal_sequence_length(train_token_ds, percentile=95)
print(f"📏 Длина последовательности (95%): {max_token}")

# === Разделение train/val ===
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

# === Построение модели ===
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

model_pwsh.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
model_pwsh.summary()

# === Callbacks ===
callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_PATH, 'best_lstm_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.TensorBoard(log_dir=LOGS_PATH)  # 🔥 Для локального просмотра в TensorBoard
]

# === Подготовка данных ===
encoder = OneHotEncoder(sparse_output=False)
y_train_enc = encoder.fit_transform(np.array(train_labels).reshape(-1, 1))
y_val_enc = encoder.transform(np.array(val_labels).reshape(-1, 1))

train_dataset = tf.data.Dataset.from_tensor_slices((train_tokens, y_train_enc)).shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_tokens, y_val_enc)).batch(16).prefetch(tf.data.AUTOTUNE)

# === Обучение ===
print("\n🚀 Обучение...")
history = model_pwsh.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset,
    callbacks=callbacks_list,
    verbose=1
)

# === Логирование статистики в ClearML ===
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

# === Визуализация и логирование графиков ===
val_probs = model_pwsh.predict(val_dataset, verbose=0)
val_preds = np.argmax(val_probs, axis=1)
val_labels_decoded = np.argmax(y_val_enc, axis=1)

acc = accuracy_score(val_labels_decoded, val_preds)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels_decoded, val_preds, average='weighted', zero_division=0)
roc_auc = roc_auc_score(val_labels_decoded, val_probs[:, 1]) if len(np.unique(val_labels_decoded)) > 1 else 0.5

print(f"""
📈 Результаты на валидации:
├─ Accuracy:  {acc:.4f}
├─ Precision: {precision:.4f}
├─ Recall:    {recall:.4f}
├─ F1-Score:  {f1:.4f}
└─ ROC-AUC:   {roc_auc:.4f}
""")

# График 1: Accuracy
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

# График 2: Loss
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

# График 3: Сводные метрики
metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']
values = [f1, precision, recall, roc_auc]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
plt.figure(figsize=(8, 5))
bars = plt.barh(metrics, values, color=colors)
plt.xlim(0, 1.1)
plt.xlabel('Значение')
plt.title('Сводные метрики модели')
for bar, val in zip(bars, values):
    plt.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
logger.report_matplotlib_figure(title="Model Metrics", series="Summary", figure=plt)
plt.close()

# График 4: Confusion Matrix
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

# === Сохранение финальной модели и артефактов ===
final_model_path = os.path.join(MODELS_PATH, 'final_lstm_model.keras')
model_pwsh.save(final_model_path)
print(f"💾 Модель сохранена: {final_model_path}")

# Сохранение метрик
results = {
    'accuracy': float(acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'roc_auc': float(roc_auc)
}
metrics_path = os.path.join(MODELS_PATH, 'metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    import json
    json.dump(results, f, indent=2, ensure_ascii=False)

# Загрузка артефактов в ClearML
try:
    task.upload_artifact('model', final_model_path)
    task.upload_artifact('metrics', metrics_path)
    print("📤 Артефакты загружены в ClearML")
except Exception as e:
    print(f"⚠️ Не удалось загрузить артефакты в ClearML: {e}")

# Завершение задачи
task.close()
print(f"✅ Обучение завершено! Ссылка: {task.get_output_log_web_page()}")