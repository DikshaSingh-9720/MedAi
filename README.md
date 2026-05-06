# MedAI — Automated imaging screening (demo)

Full-stack demo: **React** + **Node/Express/MongoDB** + **FastAPI** + **TensorFlow CNN**. Patients sign in, upload medical-style images, and receive a predicted label with confidence scores. **This is not a medical device** — replace models and validation before any clinical use.

## Architecture

| Layer | Stack |
|--------|--------|
| Frontend | Vite, React 18, Tailwind CSS, React Router |
| API | Express, JWT, Multer (local `uploads/`), Mongoose |
| ML | FastAPI, TensorFlow (CNN), optional trained weights in `ml-service/models/` |

## Prerequisites

- Node.js 18+
- MongoDB running locally (`mongodb://127.0.0.1:27017/medai`) or Atlas URI
- Python 3.10+ (for ML service)

## Setup

### 1. MongoDB

Start MongoDB or set `MONGODB_URI` in `server/.env` (copy from `server/.env.example`).

### 2. ML service

```bash
cd ml-service
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Optional — train demo weights on synthetic data (still not clinically valid):

```bash
python train_synthetic.py
```

Run the API:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Backend

```bash
cd server
copy .env.example .env
npm install
npm run dev
```

Defaults: port **5000**, ML at `http://127.0.0.1:8000`.

### 4. Frontend

```bash
cd client
npm install
npm run dev
```

Open **http://localhost:5173**. Register, then upload an image.

## Environment variables

**`server/.env`**

| Variable | Description |
|----------|-------------|
| `PORT` | API port (default 5000) |
| `MONGODB_URI` | Mongo connection string |
| `JWT_SECRET` | Strong secret for production |
| `ML_SERVICE_URL` | Python service URL |
| `CLIENT_ORIGIN` | CORS origin(s), comma-separated |

**`ml-service`**

- `CORS_ORIGINS` — optional, comma-separated (default `*`).

## API summary

- `POST /api/auth/register` — `{ email, password, fullName? }` → `{ token, user }`
- `POST /api/auth/login` — `{ email, password }` → `{ token, user }`
- `GET /api/reports` — Bearer JWT → list of reports
- `POST /api/reports` — Bearer JWT, `multipart/form-data` field **`image`** → creates report + runs ML

## Production notes

- Use HTTPS, strong `JWT_SECRET`, and restrict CORS.
- Swap local disk uploads for **Cloudinary** or S3; keep PHI compliance in mind.
- Train and validate CNN on real labeled data; consider Grad-CAM for explanations (future).
- Add doctor role, PDF export, and consultation hooks as needed.

## License

MIT — use at your own risk for research and education.
