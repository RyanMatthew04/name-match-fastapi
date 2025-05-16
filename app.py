from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import uvicorn

app = FastAPI()

df = None

# Endpoint: Upload Dataset
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global df
    try:
        
        df = pd.read_csv(file.file)
        len = df.shape[0]
        return {"message": f"Dataset uploaded successfully with {len} rows"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)