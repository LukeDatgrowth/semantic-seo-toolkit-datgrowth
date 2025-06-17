import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import openai
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

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
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"[*_~#>`]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def compute_metrics(embeddings, custom_centroid=None):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, np.nan)
    site_focus = np.nanmean(sim_matrix)
    centroid = np.mean(embeddings, axis=0) if custom_centroid is None else custom_centroid
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    site_radius = np.max(dists)
    return site_focus, site_radius, dists, centroid

# === Widoki ===
if view == "📊 SiteRadius & SiteFocus":
    st.title("📊 SiteRadius & SiteFocus")
    st.info("Tutaj możesz analizować rozrzut i spójność tematyczną treści jednej lub wielu domen.")

    uploaded_file = st.file_uploader("📁 Wgraj plik CSV (kolumny: URL, title, content)", type="csv")
    topic_input = st.text_input("🎯 (Opcjonalnie) Wprowadź temat główny strony lub bloga:")

    if uploaded_file and openai.api_key:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        if not {'URL', 'title', 'content'}.issubset(df.columns):
            st.error("⚠️ Plik musi zawierać kolumny: URL, title, content.")
        else:
            df["clean_text"] = df["content"].astype(str).apply(clean_text)
            df["embedding"] = df["clean_text"].apply(get_embedding)
            embeddings = np.vstack(df["embedding"].tolist())

            centroid_vector = get_embedding(topic_input) if topic_input else None
            site_focus, site_radius, dists, centroid = compute_metrics(embeddings, centroid_vector)

            tsne = TSNE(n_components=2, perplexity=5, random_state=42)
            coords = tsne.fit_transform(embeddings)

            df["x"] = coords[:, 0]
            df["y"] = coords[:, 1]
            df["dist_to_center"] = dists

            st.subheader("📈 Wyniki analizy")
            st.metric("SiteFocus (spójność)", f"{site_focus:.4f}")
            st.metric("SiteRadius (rozrzut)", f"{site_radius:.4f}")

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(df["x"], df["y"], s=50, alpha=0.8, edgecolor='k')

            tsne_centroid = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(np.vstack([centroid, embeddings]))
            cx, cy = tsne_centroid[0]
            ax.scatter(cx, cy, marker='X', s=200, color='blue', edgecolor='black', label="Centrum tematyczne")

            ax.set_title("t-SNE mapa tematyczna")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            st.pyplot(fig)

            st.subheader("🔍 Wpisy (z odległością od środka)")
            st.dataframe(df[["title", "URL", "dist_to_center"]].sort_values("dist_to_center", ascending=False))

elif view == "📌 Outliery tematyczne":
    st.title("📌 Wykrywanie outlierów")
    st.info("Znajdź wpisy blogowe, które tematycznie odbiegają od reszty Twojej zawartości.")

    uploaded_file = st.file_uploader("📁 Wgraj plik CSV z wpisami (URL, title, content)", type="csv", key="outliers")
    threshold = st.slider("📏 Próg odległości od środka (np. 95 percentyl):", 50, 99, 95)

    if uploaded_file and openai.api_key:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        if not {'URL', 'title', 'content'}.issubset(df.columns):
            st.error("⚠️ Plik musi zawierać kolumny: URL, title, content.")
        else:
            df["clean_text"] = df["content"].astype(str).apply(clean_text)
            df["embedding"] = df["clean_text"].apply(get_embedding)
            embeddings = np.vstack(df["embedding"].tolist())

            _, _, dists, centroid = compute_metrics(embeddings)
            df["dist_to_center"] = dists

            perc_val = np.percentile(dists, threshold)
            outliers_df = df[df["dist_to_center"] > perc_val].sort_values("dist_to_center", ascending=False)

            st.subheader(f"🚨 Wpisy uznane za outliery (>{threshold} percentyl)")
            st.dataframe(outliers_df[["title", "URL", "dist_to_center"]])

elif view == "🔗 Linkowanie wewnętrzne":
    st.title("🔗 Semantyczne linkowanie wewnętrzne")
    st.info("Generuj linki wewnętrzne między tematycznie najbliższymi wpisami.")

    uploaded_file = st.file_uploader("📁 Wgraj plik CSV z wpisami (URL, title, content)", type="csv", key="linking")
    top_n_links = st.slider("🔗 Liczba sugerowanych linków na wpis:", 1, 5, 3)

    if uploaded_file and openai.api_key:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        if not {'URL', 'title', 'content'}.issubset(df.columns):
            st.error("⚠️ Plik musi zawierać kolumny: URL, title, content.")
        else:
            df["clean_text"] = df["content"].astype(str).apply(clean_text)
            df["embedding"] = df["clean_text"].apply(get_embedding)
            embeddings = np.vstack(df["embedding"].tolist())

            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, -np.inf)

            suggestions = []
            for i in range(len(df)):
                similar_idx = np.argsort(sim_matrix[i])[::-1][:top_n_links]
                suggested_links = [(df.iloc[j]["title"], df.iloc[j]["URL"]) for j in similar_idx]
                suggestions.append(suggested_links)

            df["link_suggestions"] = suggestions

            st.subheader("🔗 Propozycje linkowania")
            for _, row in df.iterrows():
                st.markdown(f"**{row['title']}**  ")
                for link_title, link_url in row["link_suggestions"]:
                    st.markdown(f"→ [{link_title}]({link_url})")
                st.markdown("---")

elif view == "🚧 Content Gap":
    st.title("🚧 Content Gap Analysis")
    st.info("Porównaj swoją treść z konkurencją i wykryj luki tematyczne.")

    user_file = st.file_uploader("📁 Wgraj swoją treść (CSV: URL, title, content)", type="csv", key="gap_user")
    comp_file = st.file_uploader("📁 Wgraj treść konkurencji (CSV: URL, title, content)", type="csv", key="gap_comp")
    similarity_threshold = st.slider("🎯 Próg podobieństwa (cosine)", 0.0, 1.0, 0.75, step=0.01)

    if user_file and comp_file and openai.api_key:
        df_user = pd.read_csv(user_file, sep=None, engine="python")
        df_comp = pd.read_csv(comp_file, sep=None, engine="python")

        if not {'URL', 'title', 'content'}.issubset(df_user.columns.union(df_comp.columns)):
            st.error("⚠️ Oba pliki muszą zawierać kolumny: URL, title, content.")
        else:
            df_user["clean_text"] = df_user["content"].astype(str).apply(clean_text)
            df_user["embedding"] = df_user["clean_text"].apply(get_embedding)
            emb_user = np.vstack(df_user["embedding"].tolist())

            df_comp["clean_text"] = df_comp["content"].astype(str).apply(clean_text)
            df_comp["embedding"] = df_comp["clean_text"].apply(get_embedding)
            emb_comp = np.vstack(df_comp["embedding"].tolist())

            sim_matrix = cosine_similarity(emb_comp, emb_user)
            max_similarities = np.max(sim_matrix, axis=1)
            df_comp["max_similarity"] = max_similarities

            gaps_df = df_comp[df_comp["max_similarity"] < similarity_threshold].sort_values("max_similarity")

            st.subheader("🚧 Wykryte luki tematyczne względem Twojej treści")
            st.dataframe(gaps_df[["title", "URL", "max_similarity"]])

elif view == "📈 Monitoring w czasie":
    st.title("📈 Monitoring tematyczny w czasie")
    st.info("Obserwuj, jak zmienia się spójność i zakres tematyczny Twojej witryny w kolejnych okresach.")

    uploaded_files = st.file_uploader("📁 Wgraj pliki CSV z okresów (np. Q1.csv, Q2.csv...)", accept_multiple_files=True, type="csv")

    if uploaded_files and openai.api_key:
        focus_results = []
        radius_results = []
        periods = []

        for file in uploaded_files:
            df = pd.read_csv(file, sep=None, engine="python")
            if not {'URL', 'title', 'content'}.issubset(df.columns):
                st.warning(f"⚠️ Pominięto plik {file.name} — brak wymaganych kolumn.")
                continue

            df["clean_text"] = df["content"].astype(str).apply(clean_text)
            df["embedding"] = df["clean_text"].apply(get_embedding)
            embeddings = np.vstack(df["embedding"].tolist())

            focus, radius, _, _ = compute_metrics(embeddings)
            periods.append(file.name)
            focus_results.append(focus)
            radius_results.append(radius)

        if focus_results:
            result_df = pd.DataFrame({
                "Okres": periods,
                "SiteFocus": focus_results,
                "SiteRadius": radius_results
            })

            st.subheader("📊 Zmiany wartości SiteFocus i SiteRadius")
            st.line_chart(result_df.set_index("Okres"))
            st.dataframe(result_df)

elif view == "⚔️ Porównanie z konkurencją":
    st.title("⚔️ Porównanie strategii treści")
    st.info("Zobacz jak wypadasz na tle konkurencji pod względem spójności i rozrzutu tematycznego.")

    uploaded_files = st.file_uploader(
        "📁 Wgraj pliki CSV (Twoja domena + konkurencja, z kolumną 'Domena')",
        accept_multiple_files=True,
        type="csv"
    )

    if uploaded_files and openai.api_key:
        results = []

        for file in uploaded_files:
            df = pd.read_csv(file, sep=None, engine="python")
            if not {'URL', 'title', 'content', 'Domena'}.issubset(df.columns):
                st.warning(f"⚠️ Pominięto plik {file.name} — brak wymaganych kolumn.")
                continue

            for domain_name, group in df.groupby("Domena"):
                group["clean_text"] = group["content"].astype(str).apply(clean_text)
                group["embedding"] = group["clean_text"].apply(get_embedding)
                embeddings = np.vstack(group["embedding"].tolist())
                focus, radius, _, _ = compute_metrics(embeddings)
                results.append({"Domena": domain_name, "SiteFocus": focus, "SiteRadius": radius})

        if results:
            results_df = pd.DataFrame(results).sort_values("SiteFocus", ascending=False)

            st.subheader("📋 Porównanie domen")

            # Filtry interaktywne
            selected_domains = st.multiselect("🔎 Wybierz domeny do porównania:", options=results_df["Domena"].unique(), default=list(results_df["Domena"].unique()))
            focus_range = st.slider("🎯 Zakres SiteFocus:", float(results_df["SiteFocus"].min()), float(results_df["SiteFocus"].max()), (float(results_df["SiteFocus"].min()), float(results_df["SiteFocus"].max())))
            radius_range = st.slider("📐 Zakres SiteRadius:", float(results_df["SiteRadius"].min()), float(results_df["SiteRadius"].max()), (float(results_df["SiteRadius"].min()), float(results_df["SiteRadius"].max())))

            filtered_df = results_df[
                (results_df["Domena"].isin(selected_domains)) &
                (results_df["SiteFocus"].between(*focus_range)) &
                (results_df["SiteRadius"].between(*radius_range))
            ]

            st.dataframe(filtered_df)

            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 5))
            bar_width = 0.35
            x = np.arange(len(filtered_df))

            ax.bar(x - bar_width/2, filtered_df["SiteFocus"], width=bar_width, label="SiteFocus", color="#1f77b4")
            ax.bar(x + bar_width/2, filtered_df["SiteRadius"], width=bar_width, label="SiteRadius", color="#aec7e8")

            ax.set_xticks(x)
            ax.set_xticklabels(filtered_df["Domena"], rotation=45, ha="right")
            ax.set_ylabel("Wartości")
            ax.set_title("Porównanie SiteFocus i SiteRadius")
            ax.legend()
            st.pyplot(fig)

            st.download_button(
                label="⬇️ Pobierz dane jako CSV",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name="porownanie_domen.csv",
                mime="text/csv"
            )

            export_pdf = st.checkbox("📄 Wygeneruj PDF (eksperymentalne)")
            if export_pdf:
                import io
                from fpdf import FPDF

                pdf_buf = io.BytesIO()
                fig.savefig(pdf_buf, format="png")
                pdf_buf.seek(0)

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Porównanie domen - Semantic SEO", ln=True, align="C")
                pdf.image(pdf_buf, x=10, y=30, w=180)

                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                pdf_output.seek(0)

                st.download_button(
                    label="📥 Pobierz PDF",
                    data=pdf_output,
                    file_name="porownanie_domen.pdf",
                    mime="application/pdf"
                )
