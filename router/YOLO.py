from fastapi import APIRouter

YOLO_router = APIRouter(tags=["YOLO"])

@YOLO_router("/")
async def main():
  return "no result"