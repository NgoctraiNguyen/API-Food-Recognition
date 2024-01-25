from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_label

class post_name(BaseModel):
    name: str

app = FastAPI()


# Thêm middleware CORSMiddleware để hỗ trợ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các origin (để chỉ rõ các origin cụ thể, hãy thay đổi "*")
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, OPTIONS, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các header
)

@app.post("/predict")
async def predict(post_data: post_name):
    return JSONResponse(content={"text": f'xin chao {post_data.name}'}, status_code= 200)

@app.post("/recognition")
async def recognition(file: UploadFile= File(...)):
    try:
        percent, label, output = predict_label(file)
        return JSONResponse(content={"label_predict": label, "percent_max": percent, "percent": output}, status_code= 200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code= 500)