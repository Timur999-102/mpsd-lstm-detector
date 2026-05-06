import os
import re
import numpy as np
import tensorflow as tf
from clearml import Task

# Подавляем логи TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print("🔍 Поиск последней обученной модели в ClearML...")

try:
    # Ищем последнюю запущенную задачу с нашим именем
    task = Task.get_task(
        project_name="PowerShell_Malware_Detection",
        task_name="LSTM_Remote_Training"
    )

    print(f"✅ Найдена задача: {task.id}")
    print("📥 Скачивание модели (артефакта)...")

    # Автоматически скачиваем артефакт 'model' во временную папку (или берем из кэша)
    model_path = task.artifacts['model'].get_local_copy()
    print(f"✅ Модель загружена по пути: {model_path}\n")

except Exception as e:
    print(f"❌ Ошибка при скачивании модели из ClearML: {e}")
    print("Убедись, что обучение в Colab завершилось успешно!")
    exit(1)

# Загружаем скачанную модель
model_pwsh = tf.keras.models.load_model(model_path)


# === Функция предобработки ===
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


# === Функция предсказания ===
def predict_script(script_text):
    processed = clean_data_to_tokens([script_text])
    input_tensor = tf.constant(processed, dtype=tf.string)
    probs = model_pwsh.predict(input_tensor, verbose=0)[0]

    is_malicious = probs[1] > probs[0]
    return is_malicious, probs


# === Интерактивный цикл ===
print("=" * 70)
print("🛡️  PowerShell Malware Detector")
print("Введите код и нажмите Enter. Для выхода введите 'exit' или 'q'")
print("=" * 70 + "\n")

while True:
    try:
        user_input = input("📝 Код > ").strip()

        if user_input.lower() in ['exit', 'q', 'выход']:
            print("\n👋 Завершение работы. До свидания!")
            break

        if not user_input:
            continue

        is_malicious, probs = predict_script(user_input)

        if is_malicious:
            verdict = "🔴 ВРЕДОНОСНЫЙ"
            color_code = "31"
        else:
            verdict = "🟢 БЕЗОПАСНЫЙ"
            color_code = "32"

        print(f"\n ✅ Вердикт: \033[{color_code}m{verdict}\033[0m")
        print(f" 🎯 Уверенность: {max(probs):.2%}")
        print(f" 📊 Вероятности -> Безопасный: {probs[0]:.2%} | Вредоносный: {probs[1]:.2%}\n")

    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем. До свидания!")
        break
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}\n")