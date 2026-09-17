# Agritech Self-Supervised Cross-Domain Vision Transformers for Fine-Grained Crop Leaf Disease Severity Estimation

**Subsystem:** Member 3 — Frontend & System Integration  
**Research Project:** Agritech Self-Supervised Cross-Domain Vision Transformers for Fine-Grained Crop Leaf Disease Severity Estimation  
**Tech Stack:** React.js (JavaScript, Vite) • Node.js + Express.js • MongoDB + Mongoose • Multer • Plain CSS  

---

## Table of Contents

1. [System Overview & Member 3 Responsibilities](#1-system-overview--member-3-responsibilities)
2. [Dual AI Mode Architecture (Mock vs Real)](#2-dual-ai-mode-architecture-mock-vs-real)
3. [API Contract for Member 2 (AI Modeling Team)](#3-api-contract-for-member-2-ai-modeling-team)
4. [Folder Structure](#4-folder-structure)
5. [Environment Variables](#5-environment-variables)
6. [Installation & Setup](#6-installation--setup)
7. [How to Run the Application](#7-how-to-run-the-application)
8. [API Endpoints](#8-api-endpoints)
9. [Automated Testing Suite](#9-automated-testing-suite)
10. [Scientific & Research Honesty Policy](#10-scientific--research-honesty-policy)
11. [Future Improvements](#11-future-improvements)

---

## 1. System Overview & Member 3 Responsibilities

This platform serves as the central user interface, storage engine, and inference integration gateway for the research project. 

### Member 3 Responsibilities:
- **UI/UX & Frontend Development**: Clean, academic research dashboard (white/light background, readable typography, zero distracting glassmorphism/3D effects).
- **Image Ingestion**: Drag-and-drop file upload with client & server-side validation (JPG, JPEG, PNG, max 10MB), image preview, and specimen metadata.
- **Backend API Layer**: Express API orchestrating uploads, crop domain head routing, AI adapter forwarding, and database writes.
- **AI Model Integration**: Modular adapter (`backend/services/aiService.js`) with configurable dual-mode operation (`AI_MODE=mock` and `AI_MODE=real`).
- **Prediction & Severity Visualization**: Clear presentation of diagnosed pathogen, continuous severity percentage ($0\% - 100\%$), severity classification tier (Healthy, Mild, Moderate, Severe), and model confidence.
- **Segmentation / Attention Visualization**: Dual-panel viewer supporting **Side-by-Side**, **Overlay** (with adjustable opacity slider), **Original Specimen**, and **Mask Only**. Never displays fake SVG masks; gracefully displays honest fallback notices when a model does not provide spatial masks.
- **MongoDB Persistence**: Storing prediction records, timestamps, thumbnails, and provenance flags (`isMock: true/false`).
- **Error Handling**: Resilient handling of oversized files, unsupported formats, invalid crop targets, inference timeouts, and database downtime.
- **Testing & Quality Assurance**: Comprehensive automated test suite (`npm run test:suite`).

---

## 2. Dual AI Mode Architecture (Mock vs Real)

To ensure the web application is fully demonstrable and testable before Member 2 completes training and deployment of the Vision Transformer checkpoint, the system implements a strict dual-mode switch in `backend/.env`:

```env
AI_MODE=mock      # 'mock' for local UI/API testing; 'real' for live model server
AI_INFERENCE_URL=http://localhost:8000/predict
AI_TIMEOUT_MS=15000
```

### Flow Diagram:
```text
┌────────────────────────────────────────────────────────┐
│               React Frontend (Port 5173)               │
└──────────────────────────┬─────────────────────────────┘
                           │ POST /api/predict (FormData)
                           ▼
┌────────────────────────────────────────────────────────┐
│            Express Backend API (Port 5000)             │
│   ├── Multer (File Validation & Storage in uploads/)    │
│   └── Controller (Crop Whitelist Validation)           │
└──────────────────────────┬─────────────────────────────┘
                           │ Passes image & crop
                           ▼
┌────────────────────────────────────────────────────────┐
│       AI Service Adapter (services/aiService.js)       │
└──────────────┬───────────────────────────┬─────────────┘
               │ If AI_MODE=mock           │ If AI_MODE=real
               ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────────────────┐
│  Simulated Development Data  │ │ Member 2 Vision Transformer Service  │
│  - isMock: true              │ │ (FastAPI / PyTorch / ONNX on Port 8000)
│  - segmentationMask: null    │ │ - isMock: false                      │
│  (Zero fake SVG generation!) │ │ - Real lesion mask / attention map   │
└──────────────┬───────────────┘ └─────────────────┬────────────────────┘
               │                                   │
               └─────────────────┬─────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────┐
│            MongoDB Database (Predictions)              │
│       Stores crop, disease, severity, isMock flag      │
└────────────────────────────────────────────────────────┘
```

- **In Mock Mode (`AI_MODE=mock`)**:
  - The UI displays an explicit warning banner: `⚠️ [DEMO / MOCK INFERENCE]`.
  - The history table tags records with `MOCK`.
  - The Segmentation Viewer displays: `Segmentation / Attention Heatmap Unavailable` and locks overlay mode without fabricating artificial SVG shapes.
- **In Real Mode (`AI_MODE=real`)**:
  - Streams the uploaded leaf image directly to `AI_INFERENCE_URL`.
  - Tags records as `🟢 [REAL AI INFERENCE]`.
  - If Member 2's Python server is offline, returns a user-friendly error:  
    `"AI inference service is currently offline at http://localhost:8000/predict. Please ensure Member 2's Vision Transformer microservice is running."`

---

## 3. API Contract for Member 2 (AI Modeling Team)

When Member 2 provides the trained Vision Transformer model (running via FastAPI, Flask, or TorchServe):

### 1. Request Sent by Express to AI Microservice:
- **Method**: `POST`
- **URL**: `${AI_INFERENCE_URL}` (e.g. `http://localhost:8000/predict`)
- **Content-Type**: `multipart/form-data`
- **Body Fields**:
  - `image`: Binary image file (RGB JPEG/PNG)
  - `crop`: String (One of: `Tomato`, `Potato`, `Rice`, `Wheat`, `Maize`)

### 2. Expected Response from AI Microservice:
```json
{
  "crop": "Tomato",
  "disease": "Late Blight",
  "severity": 34.5,
  "confidence": 0.92,
  "severityLevel": "Moderate",
  "segmentationMask": "data:image/png;base64,iVBORw0KGgo...",
  "maskType": "attention_heatmap"
}
```

### Contract Specifications:
- `crop` *(string)*: Confirmed crop species.
- `disease` *(string)*: Diagnosed condition (e.g., `'Late Blight'`, `'Early Blight'`, `'Healthy Leaf'`).
- `severity` *(float)*: Continuous percentage value between `0.0` and `100.0`.
- `confidence` *(float)*: Softmax probability between `0.0` and `1.0`.
- `severityLevel` *(string, optional)*: Categorical tier (`'Healthy'`, `'Mild'`, `'Moderate'`, `'Severe'`). If omitted, Express calculates it automatically based on research thresholds.
- `segmentationMask` *(string or null)*: Base64 data URI (`data:image/png;base64,...`) or static URL to the 2D spatial mask. If the model head only outputs classification and severity regression without spatial attention, set to `null`.
- `maskType` *(string or null)*: `'attention_heatmap'` (continuous ViT attention weights) or `'segmentation_mask'` (binary lesion contour).

---

## 4. Folder Structure

```text
PlantDetect/
├── frontend/                             # React Frontend (JavaScript, Vite)
│   ├── public/
│   │   └── sample-leaves/                # Built-in sample leaf images for testing
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx                # Academic header with live status indicators
│   │   │   ├── CropSelector.jsx          # Crop species dropdown (5 target crops)
│   │   │   ├── ImageUpload.jsx           # Drag & drop upload with preview & clear
│   │   │   ├── ResultCard.jsx            # Honest results card with provenance banner
│   │   │   ├── SeverityDisplay.jsx       # Quantitative severity gauge & scale ticks
│   │   │   ├── SegmentationViewer.jsx    # Dual / Overlay mask visualizer
│   │   │   └── HistoryList.jsx           # MongoDB records table with Mock/Real badges
│   │   ├── pages/
│   │   │   ├── Home.jsx                  # Main analysis dashboard
│   │   │   ├── Results.jsx               # Dedicated specimen inspector
│   │   │   └── History.jsx               # Stored predictions management
│   │   ├── services/
│   │   │   └── api.js                    # Frontend API integration layer
│   │   ├── App.jsx                       # Application coordinator
│   │   ├── App.css                       # Clean academic styling
│   │   ├── index.css                     # Global reset & CSS variables
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js                    # Vite setup with /api and /uploads proxy
│   └── package.json
│
├── backend/                              # Node.js + Express Backend
│   ├── controllers/
│   │   └── predictController.js          # Controllers for prediction and history
│   ├── middleware/
│   │   ├── uploadMiddleware.js           # Multer configuration (JPG/PNG, < 10MB)
│   │   └── errorHandler.js               # Centralized error handler
│   ├── models/
│   │   └── Prediction.js                 # Mongoose schema for MongoDB
│   ├── routes/
│   │   └── predictRoutes.js              # Express API routes
│   ├── services/
│   │   └── aiService.js                  # Modular AI ViT adapter (Mock vs Real)
│   ├── uploads/                          # Uploaded image directory
│   ├── test-suite.js                     # Automated integration test suite
│   ├── test-predict.js                   # Node.js API direct test script
│   ├── .env                              # Server environment variables
│   ├── .env.example
│   ├── package.json
│   └── server.js                         # Express entrypoint & MongoDB connection
│
├── package.json                          # Root orchestration scripts
└── README.md                             # Comprehensive project documentation
```

---

## 5. Environment Variables

Configure `backend/.env` (a template is provided in `backend/.env.example`):

```env
# Server Port
PORT=5000

# MongoDB Connection String
MONGODB_URI=mongodb://127.0.0.1:27017/plant_disease_db

# Runtime Environment
NODE_ENV=development

# AI Inference Mode: 'mock' (default) or 'real'
AI_MODE=mock

# Member 2's Python Inference Server Endpoint (when AI_MODE=real)
AI_INFERENCE_URL=http://localhost:8000/predict

# Timeout for Deep Learning Inference (in milliseconds)
AI_TIMEOUT_MS=15000
```

---

## 6. Installation & Setup

### Prerequisites:
- **Node.js**: v18+ (tested on Node v24.12.0)
- **MongoDB**: Local MongoDB instance running on `127.0.0.1:27017` (or MongoDB Atlas URI)

### Install Dependencies:
From the project root directory:
```bash
npm run install:all
```
*(Installs both `backend/` and `frontend/` dependencies in one command).*

---

## 7. How to Run the Application

### Running Both Servers from Root:
Open two terminal windows:

```bash
# Terminal 1 (Backend API on http://localhost:5000):
npm run dev:backend

# Terminal 2 (Frontend App on http://localhost:5173):
npm run dev:frontend
```

### Access Points:
- **Web Dashboard**: `http://localhost:5173/`
- **Backend API**: `http://localhost:5000`
- **Health Check**: `http://localhost:5000/api/health`

---

## 8. API Endpoints

| Method | Endpoint | Description | Payload / Query |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Checks server, MongoDB, and AI service mode | None |
| `POST` | `/api/predict` | Uploads leaf image and returns prediction | `multipart/form-data` (`image`, `crop`) |
| `GET` | `/api/predictions` | Retrieves previous predictions from MongoDB | None |
| `DELETE` | `/api/predictions/:id` | Deletes a prediction record from MongoDB | URL param `:id` |
| `GET` | `/uploads/:filename` | Serves uploaded leaf image statically | None |

---

## 9. Automated Testing Suite

To verify the integrity of the upload pipeline, crop validation, mock mode provenance, and database CRUD operations:

```bash
npm run test:suite
```

Output of automated test suite:
```text
===========================================================
🌱 Starting Member 3 Integration Test Suite...
📡 Target Backend: http://localhost:5000
===========================================================

✅ TEST 1 PASSED: GET /api/health reported online, MongoDB connected, and AI mock mode.
✅ TEST 2 PASSED: POST /api/predict correctly generated mock result with isMock:true & segmentationMask:null.
✅ TEST 3 PASSED: Missing leaf image correctly rejected with HTTP 400.
✅ TEST 4 PASSED: Non-whitelisted crop correctly rejected with HTTP 400.
✅ TEST 5 PASSED: GET /api/predictions retrieved records from MongoDB.
✅ TEST 6 PASSED: DELETE /api/predictions/:id cleaned up test record.

===========================================================
🎯 Test Summary: 6 Passed, 0 Failed
===========================================================
```

---

## 10. Scientific & Research Honesty Policy

In accordance with academic standards:
1. **No Fake Neural Outputs**: The application **does not** generate synthetic SVG shapes or pseudo-random bounding boxes masquerading as Vision Transformer attention maps. When a mask is not returned by the model, `segmentationMask` is explicitly `null`, and the UI displays an honest callout box.
2. **Clear Provenance**: All mock/demo results are prominently badged in the UI and tagged with `isMock: true` in MongoDB so evaluators understand the exact boundary between the web integration layer and the AI model checkpoint.
3. **Decoupled Architecture**: When Member 2 provides the model, toggling `AI_MODE=real` seamlessly connects the full pipeline without modifying frontend code or Express routes.

---

## 11. Future Improvements

1. **Direct ONNX Runtime Node.js Integration**: Run lightweight quantized Vision Transformer `.onnx` models natively inside Node.js without requiring a separate Python runtime.
2. **Multi-Leaf Batch Processing**: Allow uploading multiple leaf specimens from a single plot to calculate canopy-level disease severity index.
3. **Printable Agronomic Diagnostics PDF**: Add a one-click "Download Laboratory Report" export formatting the prediction, severity gauge, and mask into a PDF document.
