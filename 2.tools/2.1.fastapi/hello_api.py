from fastapi import FastAPI

app = FastAPI(title="mi primera api")

estudiantes = []
@app.get("/")
def root():
    return {"message": "feliz dia del estudiante", "status": "OK"}

@app.get("/saludo")
def saludo(nombre: str):
    return {"message": f"hola {nombre}! mucho gusto en conocerte!"}

@app.post("/estudiantes")
def crear_estudiante(nombre: str, apellido: str, curso: str):
    estudiantes.append(
        {"nombre": nombre, "apellido": apellido, "curso": curso}
    )
    return estudiantes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hello_api:app", reload=True, port=8008)
