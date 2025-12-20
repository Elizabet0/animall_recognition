from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import io
from ml_dev.CNN_archi import CNN
from ml_dev.data.translate import translate

app = FastAPI()

model = CNN(num_classes=10)
model.load_state_dict(
    torch.load('ml_dev/models/best_model_acc_63.4.pt')
)
model.eval()

class_names = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]

def predict_image(bytes):
    img = Image.open(io.BytesIO(bytes))
    img = img.convert('RGB').resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor() ])
    img_tensor = transform(img).unsqueeze(0) #добавить 1 для батча

    with torch.no_grad(): #разобраааааааться
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probs, 1)

    return class_idx.item(), confidence.item()

@app.get("/")
async def home():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        class_id, confid = predict_image(content)
        return {"success": True,
            "prediction": translate[class_names[class_id]],
            "confidence": round(confid * 100, 2),
            "class_index": class_id
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        await file.close() 