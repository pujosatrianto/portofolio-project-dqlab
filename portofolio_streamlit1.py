import streamlit as st

# ----- SIDEBAR -----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Me", "Projects", "Contact"])

# ----- HEADER GLOBAL -----
st.title("👋 Hi, I'm [Nama Kamu]")
st.write("This is my interactive portfolio built with Streamlit.")

# ----- PAGE: ABOUT -----
if page == "About Me":
    st.header("About Me")
    st.write("""
    - 🛠 Background: Electrical / SCADA / Data Science
    - 🔭 Fokus riset: Smart Grid, OPC UA Security, dan Analisis Bakat Minat Siswa
    - 🎯 Goal: Membangun solusi praktis untuk industri energi & pendidikan
    """)
    st.image("https://picsum.photos/400/200", caption="My work focus")

# ----- PAGE: PROJECTS -----
elif page == "Projects":
    st.header("Highlighted Projects")

    st.subheader("1. Smart Grid Monitoring via OPC UA")
    st.write("""
    System arsitektur SCADA berbasis OPC UA + Modbus TCP untuk monitoring inverter PV skala gedung.
    """)

    st.subheader("2. Student Talent & Interest Analytics (ABM)")
    st.write("""
    Exploratory Data Analysis + LLM interpretation untuk rekomendasi pengembangan siswa sesuai tren pekerjaan masa depan.
    """)

    with st.expander("See more details"):
        st.write("Repository private / on request 😉")

# ----- PAGE: CONTACT -----
elif page == "Contact":
    st.header("Contact Me")
    st.write("📧 email: youremail@example.com")
    st.write("🌐 LinkedIn: linkedin.com/in/yourprofile")
    st.write("🏠 Location: Indonesia (UTC+7)")
