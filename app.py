import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import openai
import json
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from sqlalchemy import create_engine, text
import requests


# === Prosta autoryzacja użytkowników ===
def check_password():
    def password_entered():
        correct = st.session_state["password"] == st.secrets["APP_PASSWORD"]
        st.session_state["password_correct"] = correct
       
        
    if "password_correct" not in st.session_state:
        ip = requests.get('https://ifconfig.me/ip', timeout=5).text.strip()
        st.header(ip, divider=True)
        st.text_input("🔐 Podaj hasło", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 Podaj hasło", type="password", on_change=password_entered, key="password")
        st.error("❌ Nieprawidłowe hasło")
        st.stop()

# Uruchom logowanie
check_password()

# API Key i DB
openai.api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("🔑 Podaj swój OpenAI API Key:", type="password")

# Konfiguracja połączenia z bazą danych
# W pliku .streamlit/secrets.toml umieść:
# MYSQL_USER = "twoj_user"
# MYSQL_PASSWORD = "twoje_haslo"
# MYSQL_HOST = "adres_hosta"
# MYSQL_PORT = "3306"
# MYSQL_DB = "nazwa_bazy"
# Opcjonalnie: MYSQL_URI jeśli wolisz jednolinijkowy URI
mysql_user = st.secrets["MYSQL_USER"]
mysql_password = st.secrets["MYSQL_PASSWORD"]
mysql_host = st.secrets["MYSQL_HOST"]
mysql_port = st.secrets.get("MYSQL_PORT", "3306")
mysql_db = st.secrets["MYSQL_DB"]

# Budujemy URI SQLAlchemy
mysql_uri = st.secrets.get(
    "MYSQL_URI",
    f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}?charset=utf8mb4"
)
engine = create_engine(mysql_uri, pool_pre_ping=True)

# Tworzenie tabeli, jeśli nie istnieje
with engine.begin() as conn:
    conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            domain VARCHAR(255),
            url TEXT PRIMARY KEY,
            content TEXT,
            embedding LONGTEXT
        )
        """
    ))

# Konfiguracja strony
st.set_page_config(page_title="Semantic SEO Toolkit", layout="wide")

# Sidebar nawigacja
st.sidebar.title("🔎 Wybierz funkcję")
view = st.sidebar.radio("Dostępne analizy:", [
    "📊 SiteRadius & SiteFocus",
    "📌 Outliery tematyczne",
    "🔗 Linkowanie wewnętrzne",
    "🚧 Content Gap",
    "📈 Monitoring w czasie",
    "⚔️ Porównanie z konkurencją"
])

# === Funkcje pomocnicze ===
@st.cache_data(show_spinner=False)
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    txt = soup.get_text(separator=" ")
    txt = re.sub(r"[*_~#>`]", "", txt)
    txt = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", txt)
    txt = re.sub(r"https?://\S+", "", txt)
    txt = re.sub(r"[^\w\s.,!?-]", "", txt)
    return txt.lower().strip()

# Pobiera lub tworzy embedding w DB MySQL
def fetch_embedding(url, content):
    domain = urlparse(url).netloc
    cleaned = clean_text(content)
    sel = text("SELECT content, embedding FROM embeddings WHERE url=:url")
    # Używamy transakcji dla bezpieczeństwa
    with engine.begin() as conn:
        row = conn.execute(sel, {"url": url}).fetchone()
        if row:
            stored_content, stored_emb_json = row
            if stored_content == cleaned:
                return np.array(json.loads(stored_emb_json))
        # jeżeli brak lub zmieniony content: pobierz nowe
        response = openai.embeddings.create(model="text-embedding-3-small", input=cleaned)
        emb = response.data[0].embedding
        emb_json = json.dumps(emb)
        # wstaw lub zaktualizuj (upsert)
        upsert = text(
            "INSERT INTO embeddings(domain,url,content,embedding) VALUES(:domain,:url,:content,:emb) "
            "ON DUPLICATE KEY UPDATE content=:content, embedding=:emb"
        )
        conn.execute(upsert, {"domain": domain, "url": url, "content": cleaned, "emb": emb_json})
        return np.array(emb)

@st.cache_data(show_spinner=False)
def compute_metrics(embeddings, custom_centroid=None):
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, np.nan)
    focus = np.nanmean(sim)
    centroid = np.mean(embeddings, axis=0) if custom_centroid is None else custom_centroid
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    radius = np.max(dists)
    return focus, radius, dists, centroid
# === Widoki ===
if view == "📊 SiteRadius & SiteFocus":
    st.title("📊 SiteRadius & SiteFocus")
    st.info("Analiza rozrzutu i spójności tematycznej.")
    uploaded = st.file_uploader("📁 Wgraj CSV (URL, title, content)", type="csv")
    topic = st.text_input("🎯 Temat główny (opcjonalnie)")
    algo = st.selectbox("Algorytm redukcji wymiarów:", ['t-SNE', 'UMAP'])
    perplexity = st.slider("Perplexity (tylko t-SNE):", 5, 50, 30) if algo == 't-SNE' else None

    if uploaded and openai.api_key:
        df = pd.read_csv(uploaded)
        df['clean'] = df['content'].astype(str)
        # fetch embeddings with DB support
        df['emb'] = df.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
        E = np.vstack(df['emb'].tolist())
        C = fetch_embedding("", topic) if topic else None
        focus, radius, dists, cent = compute_metrics(E, C)


        # Redukcja wymiarów i wyznaczenie współrzędnych centrum
        if cent is not None and algo == 't-SNE':
            data = np.vstack([cent, E])
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            coords_all = tsne.fit_transform(data)
            center_point = coords_all[0]
            coords = coords_all[1:]
        else:
            if algo == 'UMAP':
                reducer = UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(E)
                center_point = reducer.transform(np.array([cent]))[0] if cent is not None else None
            else:
                # TSNE without centroid
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                coords = reducer.fit_transform(E)
                center_point = None

        df[['x', 'y']] = coords
        df['dist'] = dists

        st.metric("SiteFocus", f"{focus:.4f}")
        st.metric("SiteRadius", f"{radius:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df.x, df.y, s=50, alpha=0.8, edgecolor='k')
        if center_point is not None:
            ax.scatter(center_point[0], center_point[1], marker='X', s=200, label='Centrum tematyczne')
        ax.set_title("Mapa tematyczna")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(df[['title', 'URL', 'dist']].sort_values('dist', ascending=False))

# === Widok 2: Outliery tematyczne ===
elif view == "📌 Outliery tematyczne":
    st.title("📌 Outliery tematyczne")
    st.info("Wykryj wpisy odstające tematycznie.")
    uploaded = st.file_uploader("CSV (URL,title,content)", type="csv", key='o')
    thr = st.slider("Threshold percentyl:", 50, 99, 95)

    if uploaded and openai.api_key:
        df = pd.read_csv(uploaded)
        df['clean'] = df['content'].apply(clean_text)
        df['emb'] = df.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
        E = np.vstack(df['emb'].tolist())
        _, _, dists, _ = compute_metrics(E)
        df['dist'] = dists

        perc = np.percentile(dists, thr)
        outliers = df[df['dist'] > perc]

        # Histogram rozkładu dystansów
        fig, ax = plt.subplots()
        ax.hist(dists, bins=30)
        ax.axvline(perc, color='red', linestyle='--', label=f'{thr} percentyl')
        ax.set_title('Rozkład dystansów do centroidu')
        ax.legend()
        st.pyplot(fig)

        st.dataframe(outliers[['title', 'URL', 'dist']].sort_values('dist', ascending=False))

# === Widok 3: Linkowanie wewnętrzne ===
elif view == "🔗 Linkowanie wewnętrzne":
    st.title("🔗 Semantyczne linkowanie wewnętrzne")
    st.info("Generuj linki między tematycznie najbliższymi wpisami, wykluczając ten sam URL/domenę.")
    uploaded = st.file_uploader("CSV (URL, title, content, opcjonalnie category)", type="csv", key='linking')
    top_n = st.slider("Liczba sugerowanych linków na wpis:", 1, 5, 3)

    if uploaded and openai.api_key:
        df = pd.read_csv(uploaded)
        df['clean'] = df['content'].apply(clean_text)
        df['emb'] = df.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
        E = np.vstack(df['emb'].tolist())

        sim = cosine_similarity(E)
        np.fill_diagonal(sim, -np.inf)
        domains = df['URL'].apply(lambda u: urlparse(u).netloc)

        suggestions = []
        for i in range(len(df)):
            mask = np.ones(len(df), dtype=bool)
            mask[i] = False
            mask &= domains != domains.iloc[i]
            if 'category' in df.columns:
                mask &= df['category'] != df.loc[i, 'category']
            sims = sim[i].copy()
            sims[~mask] = -np.inf
            idx = np.argsort(sims)[::-1][:top_n]
            suggestions.append([(df.iloc[j]['title'], df.iloc[j]['URL']) for j in idx])

        df['link_suggestions'] = suggestions
        for _, row in df.iterrows():
            st.markdown(f"**{row['title']}**")
            for t, u in row['link_suggestions']:
                st.markdown(f"→ [{t}]({u})")
            st.markdown('---')

# === Widok 4: Content Gap ===
elif view == "🚧 Content Gap":
    st.title("🚧 Content Gap Analysis")
    st.info("Znajdź luki tematyczne względem konkurencji.")

    user_file = st.file_uploader("Twoja treść (CSV: URL, title, content)", type="csv", key='gap_user')
    comp_file = st.file_uploader("Treść konkurencji (CSV: URL, title, content)", type="csv", key='gap_comp')
    threshold = st.slider("Próg podobieństwa (cosine):", 0.0, 1.0, 0.75, 0.01)
    top_unique = st.number_input("Liczba najbardziej unikalnych wpisów (lowest similarity):", min_value=1, max_value=20, value=5)

    if user_file and comp_file and openai.api_key:
        df_u = pd.read_csv(user_file)
        df_c = pd.read_csv(comp_file)

        # embeddingi
        df_u['clean'] = df_u['content'].apply(clean_text)
        df_u['emb'] = df_u.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
        U = np.vstack(df_u['emb'].tolist())
        df_c['clean'] = df_c['content'].apply(clean_text)
        df_c['emb'] = df_c.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
        C = np.vstack(df_c['emb'].tolist())

        sim = cosine_similarity(C, U)
        df_c['max_sim'] = sim.max(axis=1)

        # luki poniżej progu
        gaps = df_c[df_c['max_sim'] < threshold].sort_values('max_sim')
        st.subheader("🚧 Wykryte luki poniżej progu")
        st.dataframe(gaps[['title','URL','max_sim']])

        # top unikalnych
        unique = df_c.nsmallest(top_unique, 'max_sim')
        st.subheader(f"🌟 Top {top_unique} najbardziej unikalnych wpisów")
        st.dataframe(unique[['title','URL','max_sim']])

        # podsumowanie GPT
        prompt = (
            "Podsumuj w kilku punktach kluczowe tematy, których brakuje mi na podstawie następujących tytułów konkurencji:\n" +
            "\n".join(unique['title'].tolist())
        )
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content": prompt}]
        )
        st.subheader("🔑 Podsumowanie braków tematycznych")
        st.write(resp.choices[0].message.content)

# === Widok 5: Monitoring w czasie ===
elif view == "📈 Monitoring w czasie":
    st.title("📈 Monitoring tematyczny w czasie")
    st.info("Obserwuj zmiany SiteFocus/SiteRadius w kolejnych okresach.")

    files = st.file_uploader("CSV z okresów (np. Q1.csv, Q2.csv)", type="csv", accept_multiple_files=True)
    if files and openai.api_key:
        periods, f_vals, r_vals = [], [], []
        for f in files:
            df = pd.read_csv(f)
            df['clean'] = df['content'].apply(clean_text)
            df['emb'] = df.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
            E = np.vstack(df['emb'].tolist())
            focus, radius, _, _ = compute_metrics(E)
            periods.append(f.name)
            f_vals.append(focus)
            r_vals.append(radius)

        fig, ax = plt.subplots()
        ax.plot(periods, f_vals, label='SiteFocus')
        ax.plot(periods, r_vals, label='SiteRadius')
        ax.set_xlabel('Okres')
        ax.set_ylabel('Wartość')
        ax.set_title('Monitoring tematyczny')
        ax.legend()
        st.pyplot(fig)
        st.dataframe(pd.DataFrame({'Okres':periods,'SiteFocus':f_vals,'SiteRadius':r_vals}))

# === Widok 6: Porównanie z konkurencją ===
elif view == "⚔️ Porównanie z konkurencją":
    st.title("⚔️ Porównanie strategii treści")
    st.info("Porównaj swoje wartości SiteFocus/SiteRadius z konkurencją.")

    files = st.file_uploader("CSV-y (Twoja domena + konkurencja, z kolumną 'Domena')", type="csv", accept_multiple_files=True)
    if files and openai.api_key:
        results = []
        for f in files:
            df = pd.read_csv(f)
            if not {'URL','title','content','Domena'}.issubset(df.columns):
                st.warning(f"Pominięto {f.name} – brak kolumn.")
                continue
            for dom, grp in df.groupby('Domena'):
                grp['clean'] = grp['content'].apply(clean_text)
                grp['emb'] = grp.apply(lambda row: fetch_embedding(row['URL'], row['clean']), axis=1)
                E = np.vstack(grp['emb'].tolist())
                focus, radius, _, _ = compute_metrics(E)
                results.append({'Domena':dom,'SiteFocus':focus,'SiteRadius':radius})
        if results:
            res_df = pd.DataFrame(results)
            res_df = res_df.sort_values('SiteFocus', ascending=False)

            sel = st.multiselect("Wybierz domeny:", res_df['Domena'].unique(), default=res_df['Domena'].tolist())
            r1 = st.slider("Zakres SiteFocus:", float(res_df['SiteFocus'].min()), float(res_df['SiteFocus'].max()), (float(res_df['SiteFocus'].min()), float(res_df['SiteFocus'].max())))
            r2 = st.slider("Zakres SiteRadius:", float(res_df['SiteRadius'].min()), float(res_df['SiteRadius'].max()), (float(res_df['SiteRadius'].min()), float(res_df['SiteRadius'].max())))
            filt = res_df[res_df['Domena'].isin(sel) & res_df['SiteFocus'].between(*r1) & res_df['SiteRadius'].between(*r2)]
            st.dataframe(filt)

            fig, ax = plt.subplots(figsize=(10,5))
            x = np.arange(len(filt))
            ax.bar(x-0.2, filt['SiteFocus'], width=0.4, label='SiteFocus')
            ax.bar(x+0.2, filt['SiteRadius'], width=0.4, label='SiteRadius')
            ax.set_xticks(x)
            ax.set_xticklabels(filt['Domena'], rotation=45, ha='right')
            ax.set_ylabel('Wartość')
            ax.set_title('Porównanie domen')
            ax.legend()
            st.pyplot(fig)

            # Eksport CSV
            st.download_button('⬇️ Pobierz CSV', filt.to_csv(index=False).encode('utf-8'), 'porownanie.csv','text/csv')

            # Eksport PDF
            if st.checkbox('📄 Generuj PDF'): 
                import io
                from fpdf import FPDF
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', size=12)
                pdf.cell(200,10, txt='Porównanie domen - Semantic SEO', ln=True, align='C')
                pdf.image(buf, x=10, y=30, w=180)
                outbuf = io.BytesIO()
                pdf.output(outbuf)
                outbuf.seek(0)
                st.download_button('📥 Pobierz PDF', outbuf, 'porownanie.pdf', 'application/pdf')
                st.success('PDF wygenerowano pomyślnie!')
