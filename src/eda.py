import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Dataset

print("📥 Получение датасета из ClearML...")
# Берем данные из локального кэша ClearML
dataset = Dataset.get(
    dataset_project="PowerShell_Malware",
    dataset_name="MPSD_Dataset_v2"
)
data_path = dataset.get_local_copy()

# === Функция чтения файлов ===
def read_lengths(path):
    lengths = []
    if not os.path.exists(path): return lengths
    for f in os.listdir(path):
        if not f.endswith('.ps1'): continue
        try:
            with open(os.path.join(path, f), encoding='utf-8', errors='ignore') as file:
                # Просто считаем количество слов (разделенных пробелами)
                words = file.read().split()
                lengths.append(len(words))
        except:
            pass
    return lengths

print("🔍 Подсчет длин скриптов...")
# Считаем длины вредоносных
len_bad_pure = read_lengths(os.path.join(data_path, 'malicious_pure'))
len_bad_mixed = read_lengths(os.path.join(data_path, 'mixed_malicious'))
len_malicious = len_bad_pure + len_bad_mixed

# Считаем длины чистых
len_benign = read_lengths(os.path.join(data_path, 'powershell_benign_dataset'))

print(f"📊 Обработано: {len(len_malicious)} вредоносных, {len(len_benign)} чистых.")

# === Отрисовка графика ===
# Отсекаем аномально гигантские скрипты по 95 перцентилю, чтобы график был читаемым
max_len_to_plot = int(np.percentile(len_malicious + len_benign, 95))

plt.figure(figsize=(10, 6))

# Рисуем гистограммы
sns.histplot(len_benign, bins=50, binrange=(0, max_len_to_plot),
             color='#2ecc71', label='Benign (Чистые)', stat='density', alpha=0.6, edgecolor=None)
sns.histplot(len_malicious, bins=50, binrange=(0, max_len_to_plot),
             color='#e74c3c', label='Malicious (Вредоносные)', stat='density', alpha=0.6, edgecolor=None)

# Наводим красоту
plt.title('Распределение длины PowerShell скриптов', fontsize=16, pad=15)
plt.xlabel('Длина скрипта (количество слов)', fontsize=12)
plt.ylabel('Плотность (Доля от общего числа)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Убираем рамки сверху и справа
sns.despine()
plt.tight_layout()

# Сохраняем в корень проекта
output_image = 'script_length_distribution.png'
plt.savefig(output_image, dpi=300)
print(f"✅ График успешно сохранен: {output_image}")

# Показываем на экране
plt.show()