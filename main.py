"""
Image Captioning API for Visually Impaired Assistance
Using BLIP (Bootstrapping Language-Image Pre-training) model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import torch
import uvicorn

app = FastAPI(title="Image Captioning API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained BLIP model
print("Loading BLIP model...")
# image ko resize kar taa hai, tensors me convert kar taa hai and texts to tokens me change kartaa hai
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# ye main model hai  jo ki image captioning kartaa hai ye pre-trained model ko download kar ta hai and usko load kartaa hai
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")


@app.get("/")
async def root():
    return {
        "message": "Image Captioning API for Accessibility",
        "endpoints": {
            "/caption": "POST - Upload image for captioning",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "BLIP", "device": device}


@app.post("/caption")
async def generate_caption(
    file: UploadFile = File(...), # user ke dwaraa upload image ko capture kar taa hai. File(...) -> ye required field bnaata hai.
    max_length: int = 50, # maximum caption size.
    detailed: bool = False # detailed caption generate karne ke liye use hota hai.
):
    """
    Generate caption for uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        max_length: Maximum length of generated caption
        detailed: If True, generates more detailed description
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Read and process image
        contents = await file.read() #uploaded image ko binary Bytes ke roop me read kar taa hai.
        image = Image.open(io.BytesIO(contents)).convert('RGB') #Bytes ko PIL image object me convert kar taa hai and fir usko i.r.t.a image ko RGB me convert karta hai. kuki BLIP ko RGB me hi input chaahiye.
        
        # Generate caption
        if detailed:
            # Conditional generation with prompt for detailed description
            text = "a detailed description of"
            inputs = processor(image, text, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_length=max_length, num_beams=5)
        else:
            # Standard captioning
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_length=max_length)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Generate alternative captions for context
        alternative_captions = []
        if detailed:
            for _ in range(2):
                out_alt = model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7
                )
                alt_caption = processor.decode(out_alt[0], skip_special_tokens=True)
                if alt_caption != caption:
                    alternative_captions.append(alt_caption)
        
        return {
            "success": True,
            "caption": caption,
            "alternative_captions": alternative_captions[:2],
            "image_size": f"{image.width}x{image.height}",
            "detailed": detailed
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)