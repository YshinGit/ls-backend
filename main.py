from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from pydantic import BaseModel
import uvicorn
import json
import os
import zipfile
import sqlite3
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

# ==============================
# 🔹 Configuración inicial
# ==============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Directorios
LOCAL_STORAGE = "local_storage"
MODELS_DIR = "models"
DB_FILE = "sign_language.db"

os.makedirs(LOCAL_STORAGE, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("✅ Directorios creados:", os.listdir("."))

# ==============================
# 🔹 Seguridad y DB
# ==============================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
        """)
        conn.commit()
        conn.close()
        print("✅ Base de datos inicializada")
    except Exception as e:
        print(f"❌ Error inicializando DB: {e}")

# ==============================
# 🔹 Modelos Pydantic
# ==============================
class CaptureRequest(BaseModel):
    label: str
    frames_data: list

class PredictRequest(BaseModel):
    frames_data: list

# ==============================
# 🔹 Registro y Login
# ==============================
@app.post("/register")
async def register(username: str, password: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        role = "admin" if count == 0 else "user"
        hashed_pwd = pwd_context.hash(password)
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed_pwd, role)
        )
        conn.commit()
        conn.close()
        return {"message": "Usuario registrado", "role": role}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="El usuario ya existe")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al registrar: {str(e)}")

@app.post("/login")
async def login(username: str, password: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        if not pwd_context.verify(password, user["password"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        return {"message": "Login exitoso", "role": user["role"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en login: {str(e)}")
    
# ==============================
# 🔹 Gestión de Usuarios (solo admin)
# ==============================
@app.get("/users")
async def list_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role FROM users")
        users = cursor.fetchall()
        conn.close()
        return {"users": [dict(u) for u in users]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo usuarios: {str(e)}")


@app.post("/users/set_role")
async def set_role(username: str, role: str):
    if role not in ["admin", "user"]:
        raise HTTPException(status_code=400, detail="Rol inválido")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        conn.commit()
        conn.close()
        return {"message": f"Rol de {username} actualizado a {role}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando rol: {str(e)}")


# ==============================
# 🔹 MediaPipe setup
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ==============================
# 🔹 Funciones auxiliares
# ==============================
def compress_old_files():
    files = os.listdir(LOCAL_STORAGE)
    if len(files) > 50:
        old_files = sorted(files)[:-20]
        for file in old_files:
            os.remove(os.path.join(LOCAL_STORAGE, file))

# ==============================
# 🔹 Endpoints de backend
# ==============================
@app.get("/")
async def root():
    return {"message": "✅ Backend de Lenguaje de Señas funcionando"}

@app.get("/status")
async def status():
    return {
        "capturas": len(os.listdir(LOCAL_STORAGE)),
        "modelos": len(os.listdir(MODELS_DIR)),
        "storage_path": os.path.abspath(LOCAL_STORAGE)
    }

# ==============================
# 🔹 Captura
# ==============================
@app.post("/capture")
async def capture_gesture(request: CaptureRequest):
    label = request.label
    frames_data = request.frames_data

    if len(frames_data) < 1:
        return {"error": "Debes enviar al menos 1 frame"}

    # ✅ Guardamos todos los frames recibidos
    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(LOCAL_STORAGE, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "label": label,
            "frames": frames_data,  # 🔹 guardamos todos los frames
            "timestamp": datetime.now().isoformat(),
            "frame_count": len(frames_data)
        }, f, indent=2)

    compress_old_files()
    return {
        "message": f"Captura guardada con {len(frames_data)} frames",
        "file": filename,
        "frames": len(frames_data)
    }


@app.get("/captures")
async def list_captures():
    return {"files": os.listdir(LOCAL_STORAGE)}

@app.delete("/captures/clear")
async def clear_captures():
    for f in os.listdir(LOCAL_STORAGE):
        os.remove(os.path.join(LOCAL_STORAGE, f))
    return {"message": "Todas las capturas eliminadas"}

@app.delete("/capture/{filename}")
async def delete_capture(filename: str):
    path = os.path.join(LOCAL_STORAGE, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"{filename} eliminado"}
    else:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

# ==============================
# 🔹 Entrenamiento con Sklearn
# ==============================
@app.post("/train/sklearn")
async def train_model_sklearn():
    X, y = [], []
    files = os.listdir(LOCAL_STORAGE)
    if not files:
        return {"error": "No hay capturas para entrenar"}

    for filename in files:
        if filename.endswith('.json'):
            with open(os.path.join(LOCAL_STORAGE, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data['frames']:
                    # ✅ usar todos los frames como muestras independientes
                    for frame in data['frames']:
                        flat_frame = np.array(frame).flatten().tolist()
                        X.append(flat_frame)
                        y.append(data['label'])

    if len(set(y)) < 2:
        return {"error": "Se necesitan al menos 2 clases diferentes"}

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODELS_DIR, "model_sklearn.pkl")
    joblib.dump(model, model_path)

    accuracy = np.mean(model.predict(X) == y)
    return {
        "message": "Modelo Sklearn entrenado",
        "accuracy": round(accuracy, 4),
        "classes": list(set(y)),
        "samples": len(X)  # 🔹 info adicional
    }


# ==============================
# 🔹 Entrenamiento con Keras
# ==============================
@app.post("/train/keras")
async def train_model_keras(epochs: int = 20, batch_size: int = 16, validation_split: float = 0.2):
    X, y = [], []
    files = os.listdir(LOCAL_STORAGE)
    if not files:
        return {"error": "No hay capturas para entrenar"}

    for filename in files:
        if filename.endswith('.json'):
            with open(os.path.join(LOCAL_STORAGE, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data['frames']:
                    # ✅ usar todos los frames como muestras independientes
                    for frame in data['frames']:
                        flat_frame = np.array(frame).flatten().tolist()
                        X.append(flat_frame)
                        y.append(data['label'])

    if len(set(y)) < 2:
        return {"error": "Se necesitan al menos 2 clases diferentes"}

    X = np.array(X, dtype=np.float32)
    labels = list(set(y))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    y_encoded = keras.utils.to_categorical(y_encoded, num_classes=len(labels))

    # Guardar mapeo de clases
    class_mapping_path = os.path.join(MODELS_DIR, "class_mapping.json")
    with open(class_mapping_path, 'w') as f:
        json.dump(label_to_idx, f)

    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(labels), activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X, y_encoded,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0
    )

    model_path = os.path.join(MODELS_DIR, "model_keras.h5")
    model.save(model_path)

    return {
        "message": "Modelo Keras entrenado",
        "accuracy": float(history.history["accuracy"][-1]),
        "classes": labels,
        "samples": len(X),  # 🔹 info adicional
        "history": {
            "loss": [float(l) for l in history.history["loss"]],
            "accuracy": [float(a) for a in history.history["accuracy"]]
        }
    }


# ==============================
# 🔹 Predicción
# ==============================
@app.post("/predict")
async def predict_gesture(request: PredictRequest, framework: str = "sklearn"):
    try:
        frames_data = request.frames_data
        if len(frames_data) == 0:
            return {"error": "No se recibieron frames"}

        frame = frames_data[0]  # 👉 sigue usando el frame recibido (del frontend)
        if len(frame) != 63:
            return {"error": f"Frame incorrecto. Esperado 63 valores, recibido {len(frame)}"}

        flat_frame = np.array(frame).reshape(1, -1)

        if framework == "sklearn":
            model_path = os.path.join(MODELS_DIR, "model_sklearn.pkl")
            if not os.path.exists(model_path):
                return {"error": "Primero entrena el modelo con /train/sklearn"}

            model = joblib.load(model_path)
            all_probs = model.predict_proba(flat_frame)[0]
            classes = model.classes_
            all_predictions = {str(c): round(float(all_probs[i]), 4) for i, c in enumerate(classes)}

            prediction = model.predict(flat_frame)[0]
            confidence = np.max(all_probs)

            return {"prediction": str(prediction), "confidence": round(float(confidence), 4), "all_predictions": all_predictions}

        else:  # keras
            model_path = os.path.join(MODELS_DIR, "model_keras.h5")
            class_mapping_path = os.path.join(MODELS_DIR, "class_mapping.json")

            if not os.path.exists(model_path):
                return {"error": "Primero entrena el modelo con /train/keras"}

            model = keras.models.load_model(model_path)
            probs = model.predict(flat_frame, verbose=0)[0]

            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as f:
                    label_to_idx = json.load(f)
                idx_to_label = {v: k for k, v in label_to_idx.items()}

                all_predictions = {idx_to_label.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probs)}
                prediction_idx = np.argmax(probs)
                prediction = idx_to_label.get(prediction_idx, str(prediction_idx))
                confidence = np.max(probs)

                return {"prediction": prediction, "confidence": round(float(confidence), 4), "all_predictions": all_predictions}

            else:
                prediction_idx = np.argmax(probs)
                confidence = np.max(probs)
                return {"prediction": str(prediction_idx), "confidence": round(float(confidence), 4), "all_predictions": {str(i): round(float(p), 4) for i, p in enumerate(probs)}}

    except Exception as e:
        return {"error": f"Error en predicción: {str(e)}"}
    
# ==============================
# 🔹 Endpoints de modelos activos
# ==============================
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, "active_model.json")

@app.post("/models/set_active")
async def set_active_model(framework: str):
    """
    Define qué modelo se usará por defecto en predicciones (sklearn o keras).
    """
    framework = framework.lower()
    if framework not in ["sklearn", "keras"]:
        raise HTTPException(status_code=400, detail="Framework inválido. Usa 'sklearn' o 'keras'")

    with open(ACTIVE_MODEL_FILE, "w") as f:
        json.dump({"active": framework}, f)

    return {"message": f"Modelo activo establecido en {framework}"}

@app.get("/models/get_active")
async def get_active_model():
    """
    Retorna qué modelo está configurado como activo.
    """
    if not os.path.exists(ACTIVE_MODEL_FILE):
        return {"active": None, "message": "No se ha definido un modelo activo"}
    with open(ACTIVE_MODEL_FILE, "r") as f:
        data = json.load(f)
    return {"active": data.get("active", None)}

# ==============================
# 🔹 Carga manual de modelos entrenados
# ==============================
@app.post("/upload/model/{framework}")
async def upload_model(framework: str, file: UploadFile = File(...)):
    """
    Permite subir un modelo entrenado manualmente (Sklearn .pkl o Keras .h5).
    """
    framework = framework.lower()
    if framework == "sklearn":
        filename = "model_sklearn.pkl"
    elif framework == "keras":
        filename = "model_keras.h5"
    else:
        raise HTTPException(status_code=400, detail="Framework inválido. Usa 'sklearn' o 'keras'")

    save_path = os.path.join(MODELS_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    return {"message": f"Modelo {framework} cargado correctamente", "path": save_path}


# ==============================
# 🔹 Backup
# ==============================
@app.get("/backup/download")
async def download_backup():
    if not os.listdir(LOCAL_STORAGE):
        return {"error": "No hay capturas para descargar"}
    zip_path = "backup.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(LOCAL_STORAGE):
            zipf.write(os.path.join(LOCAL_STORAGE, file), file)
    return FileResponse(zip_path, filename="sign_language_backup.zip")

@app.post("/backup/upload")
async def upload_backup(file: UploadFile = File(...)):
    temp_path = "temp_backup.zip"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    with zipfile.ZipFile(temp_path, 'r') as zipf:
        zipf.extractall(LOCAL_STORAGE)
    os.remove(temp_path)
    return {"message": f"Backup restaurado. Archivos: {len(os.listdir(LOCAL_STORAGE))}"}

# ==============================
# 🔹 Descarga de modelos entrenados
# ==============================
@app.get("/download/model/sklearn")
async def download_model_sklearn():
    model_path = os.path.join(MODELS_DIR, "model_sklearn.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo SKLearn no encontrado. Entrena el modelo primero.")
    return FileResponse(model_path, filename="model_sklearn.pkl", media_type='application/octet-stream')

@app.get("/download/model/keras")
async def download_model_keras():
    model_path = os.path.join(MODELS_DIR, "model_keras.h5")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo Keras no encontrado. Entrena el modelo primero.")
    return FileResponse(model_path, filename="model_keras.h5", media_type='application/octet-stream')

@app.get("/download/models/all")
async def download_all_models():
    sklearn_path = os.path.join(MODELS_DIR, "model_sklearn.pkl")
    keras_path = os.path.join(MODELS_DIR, "model_keras.h5")

    if not os.path.exists(sklearn_path) and not os.path.exists(keras_path):
        raise HTTPException(status_code=404, detail="No hay modelos entrenados para descargar")

    zip_path = "models_backup.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if os.path.exists(sklearn_path):
            zipf.write(sklearn_path, "model_sklearn.pkl")
        if os.path.exists(keras_path):
            zipf.write(keras_path, "model_keras.h5")

    return FileResponse(zip_path, filename="models_backup.zip", media_type='application/zip')

@app.get("/models/status")
async def models_status():
    sklearn_exists = os.path.exists(os.path.join(MODELS_DIR, "model_sklearn.pkl"))
    keras_exists = os.path.exists(os.path.join(MODELS_DIR, "model_keras.h5"))
    return {"sklearn_available": sklearn_exists, "keras_available": keras_exists, "models_dir": MODELS_DIR}

# ==============================
# 🔹 Inicio
# ==============================
if __name__ == "__main__":
    init_database()
    print("🚀 Iniciando servidor en http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
