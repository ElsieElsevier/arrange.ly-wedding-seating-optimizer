
# Arrangely API

A Flask-based REST API for wedding and event seating optimization using **spectral clustering**.

## Project Status

This is an in-progress rebuild of a hackathon project. The API runs and its endpoints respond, but the core logic is still placeholder code:

- Spectral clustering isn't implemented yet. Seating fills tables in order and ignores the relationship graph.
- There's no way to upload guests, so every event uses the same 20 generated guests.
- `/api/vectorize`, `/api/analyze_survey`, and the graph's friend edges return random or made-up data.
- `silhouette_score` and `couple_success_rate` are hardcoded, so don't rely on them.
- All data is kept in memory and is lost when the server restarts.
- The original frontend and `pythonMlService.js` no longer exist and need to be rebuilt.

## Overview

Arrangely API provides tools for optimizing seating arrangements by analyzing relationships between guests and applying clustering algorithms.

### Key Features

- **Guest Relationship Analysis** – Vectorize and analyze guest relationship data.
- **Graph Visualization** – Build and visualize relationship networks.
- **Seating Optimization** – Generate optimized seating arrangements using spectral clustering.
- **Arrangement Analytics** – Evaluate, score, and compare different seating configurations.

## Project Structure

```text
arrange.ly-spectral-clustering/
├── app.py                  # Main application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Python dependencies plus test tools
├── controllers/            # API route handlers and request validation
├── models/                 # In-memory data storage
├── services/               # Core business logic (not written yet)
└── tests/                  # Automated tests
```

## Getting Started

### Prerequisites

- Python 3.12 or newer (tested on 3.12 and 3.14)
- pip (Python package manager)

### Installation

**1. Create a virtual environment**

Run the following command from the project's root directory:

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
python app.py
```

The API will be available at `http://localhost:5000`. For a list of endpoints, open:

`http://localhost:5000/api/docs`

The server only accepts connections from your own computer. To change this, set the `HOST` and `PORT` environment variables.

## API Endpoints

### Vectorization Service

Handles text vectorization and guest relationship data processing.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/vectorize` | Convert text inputs into vector embeddings |
| `POST` | `/api/analyze_survey` | Process survey responses to extract relationship data |

### Graph Building Service

Builds, visualizes, and modifies guest relationship networks.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/events/{event_id}/build_graph` | Build a relationship graph from guest data |
| `GET` | `/api/events/{event_id}/graph_visualization` | Generate a visualization of the relationship graph |
| `POST` | `/api/events/{event_id}/update_graph` | Update the relationship graph with manual changes |

### Optimization Service

Runs seating optimization algorithms and evaluates arrangements.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/events/{event_id}/optimize` | Run the seating optimization algorithm |
| `GET` | `/api/events/{event_id}/optimize/{job_id}/status` | Retrieve the status of an optimization job |
| `GET` | `/api/events/{event_id}/optimize/{job_id}/results` | Retrieve completed optimization results |
| `POST` | `/api/events/{event_id}/score_arrangement` | Evaluate an existing seating arrangement |

### Analytics Service

Provides relationship network insights and arrangement quality metrics.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/events/{event_id}/relationship_analytics` | Generate relationship network analytics |
| `GET` | `/api/events/{event_id}/arrangements/{arrangement_id}/metrics` | Generate seating quality metrics |
| `POST` | `/api/events/{event_id}/compare_arrangements` | Compare multiple seating arrangements |
| `GET` | `/api/events/{event_id}/recommendations` | Generate seating arrangement recommendations |

## Frontend Integration

The Arrangely API is designed to integrate with a JavaScript-based frontend.

The original frontend managed its requests through `pythonMlService.js`. That file no longer exists, so a new frontend needs its own API client.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Development

To run the Flask application in development mode with debugging and automatic reloading enabled:

```bash
flask --app app run --debug
```

Debug mode provides:

- Automatic server reloading when source code changes.
- Detailed error messages for troubleshooting.
- An interactive debugger for local development.

**Note:** Debug mode is off by default. It lets anyone who can reach the server run code on your computer, so only use it locally and never combine it with `HOST=0.0.0.0` or a tunnel such as ngrok.
