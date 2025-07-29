from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)


# -------- Text Utilities -------- #
def normalize_text(text):
   if not isinstance(text, str):
       return ""
   text = text.lower()
   text = re.sub(r"\W+", " ", text)
   return text.strip()


def extract_resume_text(pdf_path):
   doc = fitz.open(pdf_path)
   return " ".join([page.get_text() for page in doc])


# -------- Load & Prepare Data -------- #
try:
   csv_path = os.path.join(os.path.dirname(__file__), "postings_clean.csv")
   postings = pd.read_csv(csv_path)


   str_cols = postings.select_dtypes(include="object").columns
   postings[str_cols] = postings[str_cols].fillna("")
   postings["job_text"] = (
       postings["title"] + " " +
       postings["description"] + " " +
       postings["skills_desc"] + " " +
       postings["skill_name"] + " " +
       postings["industry_name"]
   )
   postings["job_text"] = postings["job_text"].apply(normalize_text).astype(str)


   vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
   job_vectors = vectorizer.fit_transform(postings["job_text"])


   print("✅ Job postings loaded and vectorized:", postings.shape)


except Exception as e:
   print("❌ Failed to load job postings:", str(e))
   postings = pd.DataFrame()
   job_vectors = None


# -------- Matching Route -------- #
@app.route("/match", methods=["POST"])
def match_resume():
   try:
       file = request.files["resume"]
       file.save("uploaded_resume.pdf")
       print("✅ File received:", file.filename)


       resume_text = extract_resume_text("uploaded_resume.pdf")
       print("📄 Resume text length:", len(resume_text))


       resume_text = normalize_text(resume_text)
       resume_text = str(resume_text)


       resume_vector = vectorizer.transform([resume_text])
       scores = cosine_similarity(resume_vector, job_vectors).flatten()
       postings["match_score"] = scores


       top_matches = postings.sort_values(by="match_score", ascending=False).head(5)


       resume_words = set(re.findall(r'\b\w+\b', resume_text))
       matched_keywords = set()
       suggested_keywords = set()


       for _, row in top_matches.iterrows():
           job_text = " ".join([
               str(row.get("title", "")),
               str(row.get("skills_desc", "")),
               str(row.get("skill_name", ""))
           ]).lower()
           job_words = set(re.findall(r'\b\w+\b', job_text))
           matched_keywords.update(resume_words & job_words)
           suggested_keywords.update(job_words - resume_words)


       print("✅ Matching complete.")


       return jsonify({
           "matches": top_matches[["title", "company_name", "location", "job_posting_url", "match_score"]].to_dict(orient="records"),
           "matched_keywords": sorted(matched_keywords),
           "suggested_keywords": sorted(suggested_keywords)[:10]
       })


   except Exception as e:
       print("❌ Error in /match:", str(e))
       return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
   app.run(debug=True)