import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

# استيراد مكتبة Grad-CAM الحقيقية
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. إعدادات المعالجة والبيانات (Hyperparameters & Setup)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مسارات البيانات (يجب تعديلها لتطابق مسار MVTec على جهازك)
# هيكل المجلدات المفترض:
# data_dir/train/good , data_dir/train/defect
DATA_DIR = "./mvtec_data/pcb_dataset" 

# 2. تحضير وتحويل الصور (Data Augmentation & Normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. بناء النموذج (ResNet-34)
def build_model():
    model = models.resnet34(pretrained=True)
    # تعديل الطبقة الأخيرة لتصنيف فئتين (سليم / معيب)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model.to(DEVICE)

# 4. دالة التدريب (Training Loop)
def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    print("Starting Training Process...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {epoch_acc:.2f}%")
    
    print("Training Completed.")
    return model

# 5. تطبيق خوارزمية Grad-CAM الحقيقية
def apply_gradcam(model, image_path, target_class=1):
    model.eval()
    
    # تحديد الطبقة المستهدفة (آخر طبقة التفافية في ResNet-34)
    target_layers = [model.layer4[-1]]
    
    # إعداد أداة Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # قراءة الصورة وتجهيزها
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255
    
    # تحويل الصورة إلى Tensor يتقبله النموذج
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)
    
    # توليد الخريطة الحرارية للفئة المستهدفة (1 = معيب)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # دمج الخريطة الحرارية مع الصورة الأصلية
    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    # عرض النتائج
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(rgb_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap (Defect Explanation)")
    plt.imshow(cam_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 6. التشغيل الرئيسي (Main Execution)
if __name__ == "__main__":
    print("--- Explainable Defect Classification with Grad-CAM ---")
    
    # إعداد النموذج
    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    try:
        # محاولة تحميل البيانات (ستفشل إذا لم تقم بتوفير مجلدات البيانات)
        train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", data_transforms['train'])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # تدريب النموذج
        model = train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)
        
        # اختبار Grad-CAM على صورة معيبة (قم بتغيير المسار لصورة حقيقية من MVTec)
        sample_defect_image = f"{DATA_DIR}/test/defect/sample_01.png"
        apply_gradcam(model, sample_defect_image, target_class=1)
        
    except FileNotFoundError:
        print("\n[WARNING] Dataset not found at the specified path.")
        print("Please ensure you have downloaded the MVTec dataset and arranged it in 'train' and 'test' folders.")
        print("Skipping training. Generating Grad-CAM with an empty dummy image to demonstrate the pipeline...")
        
        # إنشاء صورة وهمية فقط لتوضيح عمل الكود البرمجي في حال غياب البيانات
        dummy_img_path = "dummy_test.png"
        cv2.imwrite(dummy_img_path, np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        apply_gradcam(model, dummy_img_path, target_class=1)