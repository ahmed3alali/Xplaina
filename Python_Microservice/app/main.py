from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schema import RecommendRequest, RecommendResponse
from .models import generate_recommendations

app = FastAPI(title="XAI Recommender Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "Python microservice running ✅"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    recs, explanations = generate_recommendations(req.user, req.top_k)
    return RecommendResponse(
        user_id=req.user.student_id,
        recommendations=recs,
        explanations=explanations
    )

if __name__ == "__main__":
    # allows: python -m app.main OR python app/main.py
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
