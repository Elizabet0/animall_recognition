import torch
print('Torch version:', torch.__version__)

# Проверка доступности CUDA
print('CUDA available:', torch.cuda.is_available())

# Создаем простую модель
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        final_size = 224 // 8
        self.fc_input_size = 64 * final_size * final_size
        self.fc = nn.Linear(self.fc_input_size, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Попробуем загрузить модель
try:
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load('ml-dev/models/best_model_acc_63.4.pt', map_location='cpu'))
    model.eval()
    print('✅ Модель загружена успешно!')
    print(f'Архитектура: {model}')
except Exception as e:
    print(f'❌ Ошибка загрузки модели: {e}')
    import traceback
    traceback.print_exc()
