import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import openai
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import io
from fpdf import FPDF
import seaborn as sns

# === Prosta autoryzacja użytkowników ===
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Podaj hasło", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 Podaj hasło", type="password", on_change=password_entered, key="password")
        st.error("❌ Nieprawidłowe hasło")
        st.stop()

# Uruchom logowanie przed całą aplikacją
check_password()

# === Cache management ===
if st.sidebar.button("🧹 Wyczyść cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# === Ustawienia API ===
openai.api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("🔑 Podaj swój OpenAI API Key:", type="password")

# === Ustawienia aplikacji ===
st.set_page_config(page_title="Semantic SEO Toolkit", layout="wide")

# === Sidebar ===
st.sidebar.title("🔎 Wybierz funkcję")
view = st.sidebar.radio("Dostępne analizy:", [
    "📊 SiteRadius & SiteFocus",
    "📌 Outliery tematyczne",
    "🔗 Linkowanie wewnętrzne",
    "🚧 Content Gap",
    "📈 Monitoring w czasie",
    "⚔️ Porównanie z konkurencją"
])

# === Wspólne funkcje ===
@st.cache_data(show_spinner=False)
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"[*_~#>`]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()

@st.cache_data(show_spinner=False)
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

@st.cache_data(show_spinner=False)
def compute_metrics(embeddings, custom_centroid=None):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, np.nan)
    site_focus = np.nanmean(sim_matrix)
    centroid = np.mean(embeddings, axis=0) if custom_centroid is None else custom_centroid
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    site_radius = np.max(dists)
    return site_focus, site_radius, dists, centroid

@st.cache_data(show_spinner=False)
def run_tsne(embeddings):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    return tsne.fit_transform(embeddings)

@st.cache_data(show_spinner=False)
def compute_similarity_matrix(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)
    return sim_matrix

@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file, sep=None, engine="python")
