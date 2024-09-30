import io
import csv
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import numpy as np
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="Files API")

@app.post("/images")
def upload_image(file: UploadFile = File(...)):
    # crear byte stream
    img_stream = io.BytesIO(file.file.read())
    # if file.content_type.split("/")[0] != "image":
    #     raise HTTPException(
    #         status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
    #         detail="Not an image"
    #     )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=415, detail=f"not supported: {e}")
    # crear array de numpy
    img_array = np.array(img_obj)
    return {"filename": file.filename, "image_size": img_array.shape}

@app.get("/reports", responses={
    200: {"content": {"text/csv": {}}}
})
def download_report() -> Response:
    data = [
        {"image": "test1.jpg", "score": 0.7},
        {"image": "test2.jpg", "score": 0.1},
        {"image": "test3.jpg", "score": 0.99},
    ]
    csv_stream = io.StringIO()
    writer = csv.DictWriter(
        csv_stream, 
        fieldnames=["image", "score"],
        quoting=csv.QUOTE_ALL
        )
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    
    text = csv_stream.getvalue()
    return Response(content=text, media_type="text/csv")
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("file_api:app", reload=True)