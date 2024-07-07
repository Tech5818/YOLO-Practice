from fastapi import APIRouter, File, UploadFile, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import os
import io
import torch
from pathlib import Path

yolo_router = APIRouter(tags=["YOLO"])
# 모델 경로 설정 (yolov5 디렉토리 내의 yolov5s.pt 파일 사용 예정)
model_path = Path(__file__).parent.parent / 'yolo' / 'yolov5'
weights_path = model_path / 'runs' / 'train' / 'result' / 'weights' / 'best.pt'
model = torch.hub.load(model_path, 'custom', path=weights_path, source='local')

def get_box(results, img):
  
    
    # 결과 이미지에 바운딩 박스와 라벨 추가
    draw = ImageDraw.Draw(img)
    for result in results.xyxy[0]:
      bbox = [
        result[0],  # xmin
        result[1],  # ymin
        result[2],  # xmax
        result[3]   # ymax
      ]  # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax) 
      if int(result[-1].item()) == 1:  # Plastic
        draw.rectangle(bbox, outline="blue", width=2)
      elif int(result[-1].item()) == 0:  # Paper
        draw.rectangle(bbox, outline="red", width=2)

      # 라벨 텍스트 추가
      label_text = model.names[int(result[-1].item())]
      draw.text((bbox[0], bbox[1]), label_text, fill="white")

    # 이미지를 바이트 스트림으로 변환하여 클라이언트에게 반환
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return img_byte_arr

# 데이터셋의 클래스 이름을 모델에 설정
model.names = ['Paper', 'Plastic']  # 실제 데이터셋의 클래스 이름으로 변경해야 함

@yolo_router.post("/predict")
async def predict(file: UploadFile):
  try:
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
  except Exception as e:
    return JSONResponse(content={"error": f"File processing error: {str(e)}"}, status_code=400)

  try:
    # 객체 탐지
    results = model(img)
    labels = [model.names[int(label)] for label in results.xyxy[0][:, -1].tolist()]

    # img_byte_arr = get_box(results, img)
    # return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
    return JSONResponse(content={"labels": labels})

  except Exception as e:
    return JSONResponse(content={"error": f"Model prediction error: {str(e)}"}, status_code=500)