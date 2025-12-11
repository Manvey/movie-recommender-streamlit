import streamlit as st
from processing import preprocess
from processing.display import Main

# ----------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Movie Matrix", page_icon="🎬")

# ----------------------------------------------------
# Modern Premium Neon Glassmorphism UI
# ----------------------------------------------------
st.markdown("""
<style>
    /* Background - Neon Purple Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0615 0%, #1a1038 50%, #0e0a1f 100%);
        background-attachment: fixed;
        color: #e2e2ff;
        font-family: 'Inter', sans-serif;
    }

    /* Glass Card (movie details card) */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(180, 120, 255, 0.28);

        /* 🔥 top padding 0 kiya, neeche wala empty box hatane ke liye */
        padding: 0 22px 22px 22px;

        /* search bar se thoda gap */
        margin-top: 16px;

        box-shadow: 0 10px 32px rgba(0,0,0,0.45);
        transition: all 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 38px rgba(0,0,0,0.6);
    }

    /* Movie small card (recommendations) */
    .movie-card {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        backdrop-filter: blur(18px);
        border: 1px solid rgba(190,150,255,0.25);
        transition: 0.25s ease-in-out;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        border: 1px solid rgba(200,150,255,0.45);
    }

    /* Titles */
    h1, h2, h3, h4 {
        font-weight: 650;
        color: #f2e6ff;
        letter-spacing: 0.6px;
    }

    /* Neon Buttons */
    .stButton button {
        background: linear-gradient(90deg, #8b5cf6, #6366f1);
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.1rem;
        color: white;
        font-weight: 700;
        box-shadow: 0px 4px 15px rgba(130, 80, 255, 0.4);
        transition: 0.3s ease;
        font-size: 16px;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #7c3aed, #4f46e5);
        transform: translateY(-3px);
        box-shadow: 0px 6px 22px rgba(150, 100, 255, 0.55);
    }

    /* Selectbox clean UI */
    .stSelectbox {
        margin-bottom: 4px !important;   /* thoda sa gap, extra space nahi */
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(200,150,255,0.25) !important;
    }

    /* 🔥 Selectbox ke andar ka text white */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #ffffff !important;      /* selected value */
    }
    .stSelectbox div[data-baseweb="select"] input {
        color: #ffffff !important;      /* jo type kar rahi ho */
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #ffffff !important;      /* placeholder / label inside */
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 **Movie Matrix - AI Movie Recommender System**")

# ----------------------------------------------------
# Session States
# ----------------------------------------------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "show_reco" not in st.session_state:
    st.session_state.show_reco = False

# ----------------------------------------------------
# Movie Details UI
# ----------------------------------------------------
def show_movie_details(name):
    info = preprocess.get_details(name)

    if info is None:
        st.error(f"Could not retrieve details for *{name}*.")
        return

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(info['poster'], use_container_width=True)

        with col2:
            st.markdown(f"## 🎥 **{name}**")
            st.markdown(f"**Release:** {info['date']} • **Runtime:** {info['runtime']} min")
            st.metric(label="IMDb Rating", value=f"⭐ {info['rating']}/10", delta=f"{info['votes']:,} votes")

            tab1, tab2, tab3 = st.tabs(["📖 Overview", "🎭 Cast", "💰 Box Office"])

            with tab1:
                st.write(info['overview'])
                st.markdown(f"**Genres:** {', '.join(info['genres'])}")
                st.markdown(f"**Director:** {info['director'][0]}")

            with tab2:
                with st.spinner("Fetching Cast..."):
                    results = preprocess.fetch_cast_parallel(info['cast_ids'])

                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(results):
                        img_url, bio = results[i]
                        cast_name = info['cast_names'][i]
                        with col:
                            st.image(img_url, use_container_width=True)
                            st.caption(f"**{cast_name}**")
                            with st.expander("Bio"):
                                st.write(bio)

            with tab3:
                st.markdown(f"**Budget:** ${info['budget']:,}")
                st.markdown(f"**Revenue:** ${info['revenue']:,}")
                profit = info['revenue'] - info['budget']
                color = "lightgreen" if profit > 0 else "salmon"
                st.markdown(
                    f"**Profit:** <span style='color:{color}'>${profit:,}</span>",
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# Recommendations UI
# ----------------------------------------------------
def show_recommendations(movie):
    st.markdown("### 🎞 **Top 5 Recommended Movies**")
    st.markdown("##### Here Are Movies Like the One You Selected:")

    paths = [
        "Files/similarity_tags_tags.pkl",
        "Files/similarity_tags_genres.pkl",
        "Files/similarity_tags_tprduction_comp.pkl",
        "Files/similarity_tags_keywords.pkl",
        "Files/similarity_tags_tcast.pkl",
    ]
    weights = [1, 1, 1, 1, 1]

    with st.spinner("Finding best matches..."):
        movies_, posters = preprocess.recommend_overall(
            new_df, movie, paths, weights=weights, top_n=5
        )

    if not movies_:
        st.info("No recommendations found.")
        return

    cols = st.columns(5)

    for i, (title, poster) in enumerate(zip(movies_, posters)):
        with cols[i]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{poster}" style="width:100%; height:250px; object-fit:cover; border-radius: 16px 16px 0 0;">
                    <div style="padding:10px; text-align:center; min-height:60px;">
                        <b style="color:white; font-size:14px;">{title}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("View Details", key=f"btn_view_{i}", use_container_width=True):
                st.session_state.selected_movie = title
                st.session_state.show_reco = False
                st.rerun()

            with st.expander("🔍 Plot Summary"):
                details = preprocess.get_details(title)
                st.caption(details['overview'] if details else "No details available.")

# ----------------------------------------------------
# Load Base Data
# ----------------------------------------------------
with Main() as bot:
    bot.main_()
    new_df, movies, movies2 = bot.getter()

# ----------------------------------------------------
# Search Bar
# ----------------------------------------------------
col_search, col_action = st.columns([3, 1])
movie_list = new_df["title"].sort_values().tolist()

try:
    default_index = (
        movie_list.index(st.session_state.selected_movie)
        if st.session_state.selected_movie
        else 0
    )
except ValueError:
    default_index = 0

with col_search:
    selected_option = st.selectbox(
        "🔍 **Search for a movie**",
        movie_list,
        index=default_index,
    )

with col_action:
    st.write("###")
    if st.button("Recommend 🔮", use_container_width=True):
        st.session_state.selected_movie = selected_option
        st.session_state.show_reco = True
        st.rerun()

if selected_option != st.session_state.selected_movie and not st.session_state.show_reco:
    st.session_state.selected_movie = selected_option

# ----------------------------------------------------
# Render Page
# ----------------------------------------------------
if st.session_state.selected_movie:
    show_movie_details(st.session_state.selected_movie)
    if st.session_state.show_reco:
        show_recommendations(st.session_state.selected_movie)
