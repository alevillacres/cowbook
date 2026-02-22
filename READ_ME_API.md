# Documentation `api_server.py` (Cowbook Processing API)

The `api_server.py` file is the core backend component of the Cowbook application. Built using **FastAPI**, it provides a RESTful interface to receive multi-camera video streams, process them through the YOLO model (for detection and tracking), and return the spatial analysis results to the frontend.

---

## 🚀 Key Features

* **Multi-Camera Processing:** Accepts and processes up to 4 video streams simultaneously.
* **Safe File Handling:** Uses temporary directories (`tempfile.TemporaryDirectory()`) to save input videos and generate output frames. This ensures that disk space is automatically freed up when the execution finishes or if an error occurs.
* **Global CORS:** Preconfigured with `allow_origins=["*"]` to accept requests from any client. This makes development and deployment via Docker much easier across local networks.
* **Hardcoded Camera Mapping:** Dynamically maps the positional indices sent by the frontend to the legacy camera IDs required by the core algorithm (Cam 1, 4, 6, 8).

---

## 📡 API Endpoints

### 1. Video Processing
`POST /`

The main endpoint that receives the video files and triggers the processing algorithm.

**Request (Content-Type: `multipart/form-data`):**
* `videos` (List[UploadFile]): The video files (MP4/AVI format) to be analyzed.
* `indices` (List[str]): The positional indices of the videos (e.g., `0`, `1`, `2`, `3`, or a comma-separated string).
* `tracking_video` (Form[str]): A flag (`"true"`/`"false"`) to request the generation of YOLO tracking videos.
* `projection_video` (Form[str]): A flag (`"true"`/`"false"`) to request the generation of the global projection video.

**Internal Logic:**
1. Maps the frontend index (0, 1, 2, 3) to the real camera ID via `CAMERA_MAPPING = [1, 4, 6, 8]`.
2. Saves the uploaded videos into a temporary folder (`/input_videos`).
3. Clones and updates the base configuration from `config.json` at runtime.
4. Calls the `process_video_group` function (imported from `group_processor.py`).
5. Reads the generated `.json` output files, distinguishing between individual camera data and the globally `merged` file.

**Success Response (`200 OK`):**
Returns a JSON object containing the status and the parsed results:
```json
{
  "status": "success",
  "results": [
    {
      "filename": "cam_1.json",
      "is_merged": false,
      "cam_id": 1,
      "tracking_video_url": "/static_videos/cam_1.avi",
      "data": { ... generated json content ... }
    },
    {
      "filename": "merged_output.json",
      "is_merged": true,
      "cam_id": null,
      "tracking_video_url": "/static_videos/merged_output.avi",
      "data": { ... global json content ... }
    }
  ]
}
