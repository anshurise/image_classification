from fastapi import FastAPI
from routes import router

app = FastAPI()

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Cat vs Dog Classification API is running!"}
