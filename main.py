from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def home():
  return "success"

import uvicorn

if __name__ == "__main__":
  uvicorn.run("main:app", host="localhost", port=8088, reload=True)