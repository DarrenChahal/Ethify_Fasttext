from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fasttext

app = FastAPI()

# Load your FastText model
model = fasttext.load_model("model_saved.ftz")  # Use your model file

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; change this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def root():
    return {"message": "FastText API is running"}

@app.post("/predict/")
def predict(text: str):
    try:
        prediction = model.predict(text)
        return {
            "text": text,
            "label": prediction[0][0],  # Top label
            "confidence": prediction[1][0],  # Confidence score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
