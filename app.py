from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5Model
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import yt_dlp
import os

app = Flask(__name__)
CORS(app)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5Model.from_pretrained("t5-base").to("cuda")
whisper = WhisperModel("base", device="cuda", compute_type="float16")

def get_t5_embedding(text):
    tokens = tokenizer(f"sentence similarity: {text}", return_tensors='pt', truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        output = model.encoder(**tokens)
    return output.last_hidden_state.mean(dim=1).cpu().numpy()

def fetch_transcript(video_id):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        return transcripts.find_transcript(['en', 'ar']).fetch()
    except:
        return None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    video_ids = data.get("video_ids", [])
    prompt = data.get("prompt", "")

    if not video_ids or not prompt:
        return jsonify({"error": "Missing parameters"}), 400

    prompt_emb = get_t5_embedding(prompt)
    results = []

    for vid in video_ids:
        transcript = fetch_transcript(vid)
        if not transcript:
            results.append({"video_id": vid, "error": "Transcript not found"})
            continue

        texts = [seg['text'] for seg in transcript]
        embeddings = np.array([get_t5_embedding(t) for t in texts])
        similarities = [cosine_similarity(prompt_emb, emb)[0][0] for emb in embeddings]
        best_idx = np.argmax(similarities)

        results.append({
            "video_id": vid,
            "best_segment": transcript[best_idx],
            "similarity": float(similarities[best_idx])
        })

    return jsonify(results)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YouTube transcript matching API is live!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
