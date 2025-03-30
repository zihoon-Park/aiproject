from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict
import uvicorn

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])

# FAST 실행명령어 자동 실행
# if __name__ == "__main__":
#     uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
