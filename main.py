from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import uvicorn
import os
from dotenv import load_dotenv
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()
api_key = os.getenv("GEMINI_API")
app = FastAPI()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Read port from env var, default 8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError if status code is 4xx/5xx
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    text = ""
    try:
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            for page in doc:
                txt = page.get_text()
                if txt.strip():
                    text += txt
                else:
                    # Handle image-based PDF using OCR
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text += pytesseract.image_to_string(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")
    
    return text

def parse_text_to_fields(text: str) -> dict:

    prompt = f"""
    Extract the following patient data in JSON format 
    with exact keys and types 
    if not available or mismatched set to "" and 
    based on the text give description and serious conditions and 
    fill respective fields: 
    birthDate: Date (YYYY-MM-DD);
    gender: 'male' | 'female' | 'other';
    address: string;
    occupation: string;
    insuranceProvider: string;
    insurancePolicyNumber: string;
    allergies: string | undefined;
    temperature: string | undefined;
    bloodPressure: string | undefined;
    description: string | undefined;
    seriousConditions: string | undefined;
    diabetes: string | undefined;
    tachycardia: string | undefined;
    hypoxia: string | undefined;
    respiratoryDistress: string | undefined;
    hypercholesterolemia: string | undefined;
    anemia: string | undefined;
    chronicKidneyDisease: string | undefined;
    hypothyroidism: string | undefined;
    hyperthyroidism: string | undefined;
    obesity: string | undefined;
    gout: string | undefined;
    coagulationDisorder: string | undefined;
    osteoporosis: string | undefined;

    Text: {text}
    """

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )

    if response.ok:
        return {"data": response.json()}
    return {"error": response.text}

@app.get("/extract-text")
def extract_report(file_url: str = Query(...)):
    try:
        text = extract_text_from_pdf(file_url)
        structured_data = parse_text_to_fields(text)
        return structured_data
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
