import logging
import json
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# Import dai moduli del tuo progetto
# Assicurati che questo file sia nella root del progetto (vicino a group_processor.py)
from group_processor import process_video_group as run_algorithm
from config_loader import load_config

# Configurazione base del logging per il server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Cowbook Processing API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000"
]
# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_VIDEO_PATH = "output_videos"
app.mount("/static_videos", StaticFiles(directory=OUTPUT_VIDEO_PATH), name="static_videos")

@app.post("/")
async def process_videos(videos:List[UploadFile] = File(...),
                         indices: List[str] = Form(...),
                         tracking_video: List[str] = Form(...),
                         projection_video: List[str] = Form(...)
                         ):
    save_tracking = str(tracking_video[0]).lower() == "true"
    save_projection = str(projection_video[0]).lower() == "true"
    print("savetrackingvideo: ", save_tracking)
    print("projectionvideo: ", save_projection)

    BASE_DIR = Path(__file__).resolve().parent
    BASE_CONFIGURATION = load_config(BASE_DIR / "config.json")
    MODEL_PATH = BASE_DIR / "models/yolov11_best.pt"

    with tempfile.TemporaryDirectory() as tmp_dir:

        input_path = Path(tmp_dir) / "input_videos"
        output_frames = Path(tmp_dir) / "output_frames"
        output_jsons = Path(tmp_dir) / "output_jsons"


        input_path.mkdir(parents=True, exist_ok=True)
        output_frames.mkdir(parents=True, exist_ok=True)
        output_jsons.mkdir(parents=True, exist_ok=True)


        dynamic_group = []

        #lista indici come string
        real_indices = []
        for item in indices:
            if "," in item:
                # Se arriva come stringa unica con virgole
                real_indices.extend(item.split(","))
            else:
                # Se arriva già separato
                real_indices.append(item)

        CAMERA_MAPPING = [1, 4, 6, 8]
        for video, index in zip(videos, real_indices):
            idx = int(index)

            # 2. Logica di Mappatura Sicura
            legacy_camera_id = 0

            if 0 <= idx < len(CAMERA_MAPPING):
                # Se è 0->1, 1->4, 2->6, 3->8
                legacy_camera_id = CAMERA_MAPPING[idx]
            # dove verrà caricato il video del utente
            dest_path = Path(input_path) / f"cam_{legacy_camera_id}.mp4"

            try:
                with open(dest_path, "wb") as buffer:
                    shutil.copyfileobj(video.file, buffer)

                dynamic_group.append({
                    "path": str(dest_path),
                    "camera_nr": legacy_camera_id
                })
                print(f"Salvataggio video: {dest_path} (Mapping: {idx} -> {legacy_camera_id})")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Errore salvataggio file: {e}")

        if not dynamic_group:
            logger.warning("No video group found")

        # --- Configurazione Run-Time ---
        # Aggiorniamo la config con i path corretti
        run_config = BASE_CONFIGURATION.copy()
        run_config["model_path"] = str(MODEL_PATH)
        run_config["output_json_folder"] = str(output_jsons)
        run_config["output_image_folder"] = str(output_frames)
        run_config["save_tracking_video"] = tracking_video
        run_config["create_projection_video"] = projection_video


        try:
            run_algorithm(
                group_idx=1,
                video_group=dynamic_group,
                model_ref=str(MODEL_PATH),
                config=run_config,
                output_json_folder=str(output_jsons),
                output_image_folder=str(output_frames),
            )

            response_data = []
            json_files = list(output_jsons.glob("*.json"))

            for j_file in json_files:
                with open(j_file, "r") as f:
                    data = json.load(f)

                    # 1. Identifichiamo se è il file "merged" (quello globale)
                    is_merged = "merged" in j_file.name

                    # 2. Estraiamo il numero della cam (se non è merged)
                    cam_id = None
                    video_name = j_file.name.replace(".json", ".avi")
                    tracking_url = f"/static_videos/{video_name}"
                    if not is_merged:
                        # Cerchiamo il numero dopo "cam_" nel nome del file
                        import re
                        match = re.search(r"cam_(\d+)", j_file.name)
                        if match:
                            cam_id = int(match.group(1))

                    # 3. Creiamo un oggetto arricchito
                    response_data.append({
                        "filename": j_file.name,
                        "is_merged": is_merged,
                        "cam_id": cam_id,  # Questo sarà 1, 4, 6 o 8
                        "tracking_video_url": tracking_url,
                        "data": data,

                    })

            return {"status": "success", "results": response_data}

        except Exception as e:
            logger.exception("Errore durante l'esecuzione dell'algoritmo: {e}")
            return {"status": "error", "results": {"error": str(e)}}