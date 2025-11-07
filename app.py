# Import Required Libraries
import pandas as pd
import numpy as np
import pickle
import io
import time
import math
import random
import seaborn as sns
import joblib
from scipy.stats import pearsonr, jarque_bera
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Page Configuration
st.set_page_config(layout="wide", page_title="Portofolio Project", page_icon=":heart:")
st.sidebar.title("Navigation")
nav_options = ["Home", "Tools dan Software", "Data Collecting", "Exploratory Data Analysis", "Data Preprocessing", "Feature Engineering", "Modelling", "Prediction", "About"]
nav = st.sidebar.selectbox("Go to", nav_options)

st.markdown(
    """
    <style>
    .justify-text {
        text-align: justify;
    }
    .center-links {
        text-align: center;
        line-height: 1.6;
    }
    .section-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Dataset Page
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
df = pd.read_csv(url)

# Function Heart Disease Prediction
def heart():
    st.write("""
        This app predicts the **Heart Disease**

        Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)
    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_manual():
            st.sidebar.header("Manual Input")
            age = st.sidebar.slider("Age", 0, 100, 25)
            cp = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
            if cp == "Typical Angina":
                cp = 1
            elif cp == "Atypical Angina":
                cp = 2
            elif cp == "Non-anginal Pain":
                cp = 3
            elif cp == "Asymptomatic":
                cp = 4
            thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 200, 100)
            slope = st.sidebar.selectbox("Slope", ("Upsloping", "Flat", "Downsloping"))
            if slope == "Upsloping":
                slope = 1
            elif slope == "Flat":
                slope = 2
            elif slope == "Downsloping":
                slope = 3
            ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
            oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 0.0)
            exang = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
            if exang == "Yes":
                exang = 1
            elif exang == "No":
                exang = 0
            thal = st.sidebar.selectbox("Thal", ("Normal", "Fixed Defect", "Reversable Defect"))
            if thal == "Normal":
                thal = 1
            elif thal == "Fixed Defect":
                thal = 2
            elif thal == "Reversable Defect":
                thal = 3
            sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
            if sex == "Male":
                sex = 1
            else:
                sex =0
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_manual()

    # Data df
    st.image('heart-disease.jpg', width=700)

    if st.sidebar.button("GO!"):
        df = input_df.copy()
        st.write(df)
        model = joblib.load('model/random_forest.joblib')
        prediction = model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else 'Heart Disease']
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success('This patient has {}'.format(result[0]))
            st.balloons()
# Home Page
if nav == "Home":
    st.title("Portofolio Project - Heart Disease Prediction")
    link_dqlab = '<a href="https://dqlab.id">DQLAB Academy</a>'
    perkenalan1 = '''
    Halo Sobat semua, kenalin nama saya Pujo Satrianto. Saya merupakan ASN di salah satu instansi pemerintah
    di Indonesia. Kali ini saya mencoba menyajikan project portofolio yang saya buat sebagai salah satu syarat
    kelulusan pada kelas Machine Learning & AI for Beginner di 
    '''
    perkenalan2 = '''Project kali ini yaitu prediksi
    penyakit jantung menggunakan algoritma Machine Learning yang diimplementasikan dengan Streamlit agar
    tampilannya interaktif dan mudah dipahami siapa saja. Semoga project sederhana ini bisa bermanfaat ya, Sobat!
    '''
    st.write('''
    **Bootcamp Machine Learning & AI for Beginner [DQLAB](https://dqlab.id)**
    ''')
    st.markdown(f'<p class="justify-text">{perkenalan1} {link_dqlab}. {perkenalan2}</p>', unsafe_allow_html=True)

    st.image("https://img.freepik.com/free-photo/top-view-world-heart-day-concept-with-stethoscope_23-2148631023.jpg?t=st=1761747441~exp=1761751041~hmac=f764d703351a223d94077c3b6653735957caf438d66ad6bd1da21c4a6f2a8ba4&w=1480",
             width=700, caption="www.freepik.com")
    
    paragraf1 = '''
    Sobat, penyakit jantung itu udah jadi momok yang serius banget di Indonesia! Berdasarkan data
    Kementerian Kesehatan terbaru tahun 2025, ternyata DIY jadi jurang teratas dengan prevalensi
    penyakit jantung mencapai 1,67%, disusul Papua Tengah (1,65%) dan DKI Jakarta (1,56%). Yang
    bikin miris, data Survei Kesehatan Indonesia (SKI) 2023 menunjukkan kelompok usia 25-34 tahun
    malah jadi yang paling banyak kena penyakit jantung dengan 140.206 orang, bahkan ngalahin
    kelompok usia lebih tua. Gaya hidup anak muda sekarang yang serba instan, sering stress, jarang
    olahraga, plus kebiasaan ngerokok dan pola makan yang sembarangan jadi biang keladinya.
    Lebih parahnya lagi, dari data BPJS Kesehatan 2023, klaim untuk penyakit jantung iskemik aja
    udah nyentuh angka 20 juta kasus dengan biaya hampir Rp17,6 triliun!
    '''

    paragraf2 = '''
    Nah, di sinilah pentingnya deteksi dini yang bisa jadi game changer untuk mencegah hal-hal yang 
    nggak diinginkan. Machine learning sebagai teknologi yang lagi hits sekarang ternyata bisa 
    dimanfaatin banget untuk bikin sistem prediksi risiko penyakit jantung yang accessible dan user- 
    friendly. Project portfolio ini bakal menggunakan dataset klasik tapi powerful, yaitu dataset Heart
    Disease, yang diimplementasikan lewat Streamlit biar tampilannya interactive dan mudah
    dipahami siapa aja. Dengan memanfaatkan algoritma machine learning, kita bisa menganalisa pola-
    pola kompleks dari parameter medis seperti tekanan darah, kolesterol, detak jantung, dan faktor
    risiko lainnya untuk memprediksi kemungkinan seseorang terkena penyakit jantung. Harapannya,
    tool ini bisa jadi screening awal yang membantu masyarakat lebih aware sama kondisi kesehatan
    jantung mereka, sekaligus nunjukin gimana teknologi AI bisa berkontribusi nyata dalam dunia
    healthcare Indonesia yang lagi butuh solusi inovatif dan terjangkau. Dataset yang digunakan adalah
    dataset penyakit jantung dari
    '''

    # Buat variabel untuk link HTML agar lebih rapi
    link_html = '<a href="https://archive.ics.uci.edu/ml/datasets/heart+Disease">UCI Machine Learning Repository</a>'

    st.write('''
    **Project Overview**
    ''')
    st.markdown(f'<p class="justify-text">{paragraf1}</p>', unsafe_allow_html=True)
    st.write('''
    [Sumber] :
    1. https://jogjapolitan.harianjogja.com/read/2025/10/10/510/1231374/diy-catat-prevalensi-penyakit-jantung-167-persen-lampaui-angka-nasional
    2. https://data.goodstats.id/statistic/pasien-jantung-di-indonesia-didominasi-usia-produktif-79yo9
    3. https://www.cnnindonesia.com/gaya-hidup/20251007131438-255-1281868/diy-catat-kasus-penyakit-jantung-tertinggi-di-indonesia
    4. https://www.rsi.co.id/informasi/artikel/hari-jantung-sedunia-2025-waktunya-kenali-penyakit-jantung-pembunuh-nomor-1-di-dunia
    ''')
    st.markdown(f'<p class="justify-text">{paragraf2} {link_html}.</p>', unsafe_allow_html=True)


    st.write('''
    **Project Objective**
    
    Tujuan dari project ini adalah untuk melakukan prediksi apakah seseorang beresiko memiliki penyakit jantung berdasarkan
    pemodelan yang telah dilakukan menggunakan Machine Learning.
    ''')

elif nav == "Tools dan Software":
    st.title("Tools dan Software yang Digunakan")
    st.write('''
    Dalam project prediksi penyakit jantung ini, beberapa tools dan software yang digunakan meliputi:
    
    1. **Python**: Bahasa pemrograman utama yang digunakan untuk analisis data, pemodelan machine learning, dan pengembangan aplikasi.
    2. **Pandas**: Library Python yang digunakan untuk manipulasi dan analisis data, termasuk pembersihan data dan eksplorasi data.
    3. **NumPy**: Library Python yang digunakan untuk komputasi numerik dan operasi array.
    4. **Scikit-learn**: Library machine learning yang digunakan untuk membangun dan mengevaluasi model prediksi penyakit jantung.
    5. **Streamlit**: Framework open-source yang digunakan untuk membuat aplikasi web interaktif untuk menampilkan hasil prediksi model.
    6. **Google Colab**: Platform berbasis cloud yang digunakan untuk menjalankan kode Python dan melakukan eksperimen machine learning tanpa perlu konfigurasi lokal.
    ''')  

elif nav == 'Data Collecting':
    st.title("Dataset Collecting")
    st.write('''
    **Dataset Overview**
    
    Jadi sob, dalam project Machine Learning pasti akan diawali dengan mengumpulkan data (Data Collecting).
    Pengumpulan data (Data Collecting) ini bisa dilakukan dengan berbagai cara mulai dari pengukuran atau
    akuisisi data secara langsung sampai dengan memakai dataset yang sudah tersedia secara global dan gratis.
    Seperti pada project kali ini, karena saya bukan seorang tenaga kesehatan maka pengumpulan data dilakukan
    dengan mengambil dataset penyakit jantung dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    Dataset ini punya 1025 baris dan 13 kolom fitur dan 1 kolom target. Kolom `target` berfungsi untuk menandai
    kondisi kesehatan jantung seseorang, di mana jika bernilai `1` berarti **terindikasi punya penyakit jantung**,
    sedangkan nilai `0` berarti **tidak terindikasi punya penyakit jantung**.  \n
    Yuk kita lihat bagaimana tampilan datasetnya di bawah ini dengan menampilkan 20 baris data teratas.
    ''')

    #show dataset head
    st.dataframe(df.head(20))

    st.write('''
    Nah, sekarang yuk kita coba pahami deskripsi dari tiap kolom/fitur dari dataset ini.
    
    1. `age` : usia dalam tahun (umur)
    2. `sex` : jenis kelamin (1 = laki-laki; 0 = perempuan)
    3. `cp` : jenis nyeri dada yang dirasakan oleh pasien dengan 4 nilai kategori yang mungkin:
        - 0: typical angina
        - 1: atypical angina
        - 2: non-anginal pain
        - 3: asymptomatic
    4. `trestbps` : tekanan darah pasien pada saat istirahat, diukur dalam mmHg (milimeter air raksa (merkuri))
    5. `chol` : kadar kolesterol serum dalam darah pasien, diukur dalam mg/dl (miligram per desiliter)
    6. `fbs` : kadar gula darah pasien saat puasa (belum makan)
        - 0 : tidak lebih dari 120 mg/dl
        - 1 : lebih dari 120 mg/dl
    7. `restecg` : hasil elektrokardiogram pasien saat istirahat dengan 3 nilai kategori
        - 0: normal
        - 1: memiliki ST-T wave abnormalitas (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        - 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes
    8. `thalach` : detak jantung maksimum yang dicapai oleh pasien selama tes olahraga, diukur dalam bpm (denyut per menit)
    9. `exang` : angina yang diinduksi oleh olahraga (1 = yes; 0 = no)
    10. `oldpeak` : seberapa banyak ST segmen menurun atau depresi saat melakukan aktivitas fisik dibandingkan saat istirahat
    11. `slope` : kemiringan segmen ST pada elektrokardiogram (EKG) selama latihan fisik maksimal dengan 3 nilai kategori
        - 1: naik
        - 2: datar
        - 3: turun
    12. `ca` : jumlah pembuluh darah utama (0-3) yang diwarnai dengan flourosopy
    13. `thal` : hasil tes thalium scan dengan 3 nilai kategori
         - 1 = normal
         - 2 = adanya cacat tetap pada thalassemia
         - 3 = cacat yang dapat perbaiki
    14. `target` : memiliki penyakit jantung atau tidak (1 = yes; 0 = no)
    ''')

    # show dataset shape
    st.write(f'''
    **Dataset Shape:** {df.shape}  \nShape ini apa sih, Sob? Jadi, shape ini maksudnya dataset ini punya {df.shape[0]}
    baris data dan {df.shape[1]} kolom.
    ''')

    # show dataset count visualization
    st.write('''
    **Dataset Count Visualization**
    
    Sekarang, kita coba lihat visualisasi dari beberapa kolom penting di dataset ini. Kamu tinggal pilih aja dari dropdown
    di bawah ini dan tunggu sebentar untuk lihat visualisasinya.
    ''')
    views = st.selectbox("Select Visualization", ("", "Target", "Age", "Age by Target"))
    if views == "Target":
        st.bar_chart(df.target.value_counts())
        st.write('''
        Seperti yang sudah dijelaskan sebelumnya pada Dataset Overview, kolom/fitur `target` jika bernilai 1
        menunjukkan seseorang terindikasi memiliki penyakit jantung, sedangkan nilai 0 menunjukkan tidak terindikasi
        memiliki penyakit jantung. Berdasarkan visualisasi di atas, Sobat bisa lihat jumlah orang yang terindikasi
        memiliki penyakit jantung sejumlah 526 orang, lebih banyak daripada yang tidak terindikasi memiliki penyakit
        jantung yaitu sejumlah 499 orang.
        ''')
    elif views == "Age":
        st.bar_chart(df['age'].value_counts())
        st.write('''
        Sobat semua bisa lihat bahwa data responden ini didominasi oleh usia 50-60 tahun. Hal ini menunjukkan bahwa
        risiko penyakit jantung cenderung meningkat seiring bertambahnya usia, terutama pada rentang usia tersebut.
        Tapi ingat ya, penyakit jantung juga bisa menyerang usia yang lebih muda tergantung pada gaya hidup dan faktor risiko lainnya.
        ''')
    elif views == "Age by Target":
        data_grup = df.groupby('age')['target'].value_counts().unstack(fill_value=0)
        st.bar_chart(data_grup)
        st.write('''
        Nah, visualisasi yang ini merupakan kombinasi antara usia dan target. Dari sini, Sobat bisa lihat bahwa
        yang terindikasi memiliki penyakit jantung tertinggi pada usia 54 tahun, Sob.  Padahal usia ini belum terlalu tua,
        lho. Bahkan Sobat bisa lihat juga justru yang berusia 29, 34, dan 37 tahun semuanya terindikasi memiliki penyakit jantung.
        Ini mengingatkan kita semua bahwa penyakit jantung bisa menyerang siapa saja, nggak peduli usia muda atau tua.
        Tapi sekali lagi, ini hanya berdasarkan data yang ada di dataset ini ya, Sobat.
        ''')  

elif nav == "Exploratory Data Analysis":
    lst = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    #df[lst] = df[lst].astype(object)

    st.header("Exploratory Data Analysis")
    st.write('''
    Sobat, ketika kita sudah memiliki data, pasti lebih lanjut kita ingin tahu lebih dalam tentang data tersebut, kan? Nah,
    Exploratory Data Analysis (EDA) ini adalah tahap di mana kita mencoba untuk memahami data lebih dalam lagi. Melalui EDA,
    kita bisa menemukan pola, tren, dan hubungan antar fitur dalam dataset. Dengan EDA, kita bisa mengidentifikasi anomali,
    outlier, atau pola yang mungkin mempengaruhi hasil prediksi model machine learning yang akan kita buat nanti. Yuk,
    kita bahas satu per satu ya, Sobat!
    ''')  

    st.write('''
    **1. Menampilkan Informasi Dataset**  
    Pada tahap awal ini kita coba menampilkan informasi dan summary statistik dari dataset ini. Hal ini penting banget sebagai
    langkah awal untuk memahami struktur data, tipe data, dan ringkasan statistik dari tiap fitur dalam dataset. Dengan memahami
    informasi dasar ini, kita bisa menentukan langkah-langkah selanjutnya dalam proses analisis data. Berikut informasi dari dataset ini:
    ''')  
    col_describe, col_info = st.columns(2)

    with col_info:
        st.write('''
        **Dataset Information**
        ''')
        # Tangkap output df.info() yang asli
        buffer_awal = io.StringIO()
        df.info(buf=buffer_awal)
        info_awal_string = buffer_awal.getvalue()
    
        # Tampilkan di kolom kiri
        st.code(info_awal_string)
    
    with col_describe:
        # show dataset description
        st.write('''
        **Statistical Summary**
        ''')
        st.dataframe(df.describe())  

    st.write("""
    Ringkasan statistik ini dihasilkan oleh perintah `df.describe()` sedangkan dataset info dihasilkan dari perintah `df.info()`. Keduanya 
    memberikan gambaran singkat mengenai distribusi dan skala setiap fitur (kolom) numerik dalam data Anda. Ini adalah alat penting untuk 
    **memahami data kita** sebelum membangun model.  
    
    Kita bisa pahami dari tabel ini:
    1. `count`: Jumlah baris data yang valid (tidak kosong). Di sini, semua kolom memiliki **1025** data. Berarti kita tidak perlu
        melakukan penanganan missing value.
    2. `mean`: Nilai rata-rata dari kolom tersebut. Contoh: Rata-rata usia (`age`) pasien adalah **54.4 tahun**.
    3. `std`: Standar Deviasi, menunjukkan seberapa tersebar data dari nilai rata-ratanya.
    4. `min`: Nilai terkecil. Contoh: Usia termuda (`age`) adalah **29 tahun**.
    5. `max`: Nilai terbesar. Contoh: Usia tertua (`age`) adalah **77 tahun**.
    6. `25%`, `50%`, `75%`: Kuartil. `50%` (Median) adalah nilai tengah data. 
    7. Kita bisa lihat 50% pasien berusia di bawah **56 tahun**.
    8. `target`: Rata-rata (`mean`) dari kolom `target` adalah **0.5132**. Karena targetnya 0 dan 1, ini berarti 
        sekitar **51.3%** pasien dalam dataset ini memiliki penyakit jantung (target=1).
    """)  

    st.write('''
    **2. Menampilkan Boxplot untuk Mendeteksi Outlier**  
    Outlier ini adalah data yang nilainya jauh berbeda dari nilai-nilai lainnya dalam dataset. Outlier ini bisa terjadi karena
    kesalahan pengukuran, kesalahan pencatatan data, atau memang data yang benar-benar berbeda. Outlier ini bisa mempengaruhi
    hasil analisis dan model machine learning yang kita buat nanti. Oleh karena itu, penting banget untuk mendeteksi dan menangani
    outlier ini. Pada project ini, kita akan menampilkan boxplot untuk melihat outlier pada data numerik. Berikut boxplotnya:
    ''')

    #Menampilkan boxplot untuk melihat outliers data numerik
    df_numeric = df.drop(columns=lst)
    df_numeric.plot(kind = 'box', subplots = True, layout = (2,7), sharex = False, sharey = False, figsize = (20, 10), color = 'k')
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

    st.write('''
    Naaah, dari boxplot di atas Sobat bisa lihat bahwa ada beberapa gambar boxplot yang memiliki titik-titik di luar boxplot.
    Titik-titik tersebut adalah outlier. Outlier ini bisa mempengaruhi hasil analisis dan model machine learning yang kita buat nanti.
    Oleh karena itu, penting banget untuk mendeteksi dan menangani outlier ini. Namun pada EDA ini, kita hanya akan menampilkan boxplot
    untuk mendeteksi outlier saja ya, Sobat.
    ''')  

    st.write('''
    **3. Memvisualisasikan Distribusi Variabel Numerikal dan Kategorikal**  
    Selanjutnya, kita akan memvisualisasikan distribusi variabel numerikal dan kategorikal pada dataset ini. Visualisasi ini
    penting banget karena membantu kita untuk memahami sebaran data, apakah data tersebut normal atau skewed. Dengan memahami
    distribusi data, kita bisa menentukan teknik feature engineering yang tepat sebelum melatih model machine learning. Berikut
    visualisasi distribusi dari tiap variabel:   
    ''')

    col_num, col_cat = st.columns(2)
    
    with col_cat:
        st.write('''
        **Distribusi Variabel Kategorikal**
        ''')
        categorical_col = df[lst]
        plt.figure(figsize=(12,11.5))
        for index, column in enumerate(categorical_col.columns):
            plt.subplot(4, 3, index+1)
            sns.countplot(data=categorical_col,x=column, hue='target', palette='magma')
            plt.xlabel(column.upper(),fontsize=14)
            plt.ylabel("count", fontsize=14)

        plt.tight_layout(pad = 1.0)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()
      
    with col_num:
        st.write('''
        **Distribusi Variabel Numerikal**
        ''')
        plt.figure(figsize=(16,12))
        for index,column in enumerate(df_numeric):
            plt.subplot(2,3,index+1)
            sns.histplot(data=df_numeric,x=column,kde=True)
            plt.xticks(rotation = 90)
        plt.tight_layout(pad = 1.0)
        fig_num = plt.gcf()  # 1. Ambil figure yang baru Anda buat
        st.pyplot(fig_num) # 2. Tampilkan di Streamlit
        plt.clf()  

    st.write('''
    Dari visualisasi di atas, Sobat bisa lihat bahwa sebagian besar variabel numerikal memiliki distribusi yang
    mendekati normal, meskipun ada beberapa yang sedikit skewed. Sedangkan untuk variabel kategorikal, distribusi frekuensi 
    dari masing-masing kategori menunjukkan bahwa beberapa kategori memiliki jumlah yang lebih dominan dibandingkan kategori 
    lainnya. Dari sini kita bisa pahami bahwa kita nanti harus melakukan penanganan khusus pada variabel-variabel yang 
    memiliki distribusi yang tidak merata ini.
    ''')  

    st.write('''
    **3. Memvisualisasikan Correlation Matrix**  
    Tahap ini kita akan memvisualisasikan correlation matrix dari dataset ini. Correlation matrix ini penting banget karena
    membantu kita untuk memahami hubungan antar fitur dalam dataset. Dengan memahami hubungan antar fitur, kita bisa menentukan
    fitur-fitur yang paling berpengaruh terhadap target variabel. Berikut correlation matrix dan grafiknya:   
    ''')  

    col_cor, col_corgraf = st.columns(2)

    with col_cor:
        st.write('''
        **Correlation Matrix**
        ''')
        cor_matrix = df.corr()
        st.dataframe(cor_matrix)

    with col_corgraf:
        st.write('''
        **Correlation Heatmap**
        ''')  
        plt.figure(figsize=(20,15))
        sns.heatmap(cor_matrix,annot=True, linewidth=.5, cmap="magma")
        plt.title('Korelasi Antar Variable', fontsize = 30)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()  
    
    st.write('''
    **Mengurutkan Correlation terhadap Target Variable**  
    Selain menampilkan correlation matrix dan heatmap, kita juga akan mengurutkan correlation terhadap target. 
    Dengan mengurutkan correlation terhadap target, kita bisa dengan mudah melihat fitur-fitur mana saja yang paling
    berpengaruh terhadap target variabel. Berikut urutan correlation terhadap target:
    ''')  
    cor_matrix['target'].sort_values()
    st.dataframe(cor_matrix['target'].sort_values().to_frame().transpose())  

    st.write('''
    Korelasi positif dengan variabel tertentu berarti semakin tinggi variabel tersebut maka akan semakin tinggi juga
    kemungkinan terkena penyakit jantung, sedangkan korelasi negatif ialah semakin rendah nilai variabel tersebut maka 
    kemungkinan terkena penyakit jantung lebih tinggi. Berikut penjelasan singkatnya, Sobat:  
      
    1. Korelasi Positif
       - `cp` : Indikator positif terkuat. Tipe nyeri dada tertentu sangat berkorelasi dengan penyakit jantung
       - `thalach` : Indikator positif kuat. Detak jantung maksimum yang tinggi berkorelasi dengan penyakit jantung
       - `slope` : Indikator positif sedang. Tipe kemiringan ST segment (slope) berkorelasi dengan penyakit jantung
       - `restecg` : Indikator positif lemah
    
    2. Korelasi Negatif
       - `oldpeak` : Indikator negatif terkuat. Nilai depresi ST (oldpeak) yang tinggi sangat berkorelasi dengan tidak 
         adanya penyakit jantung
       - `exang` : Indikator negatif kuat. Mengalami angina akibat olahraga (exang=1) sangat berkorelasi dengan tidak 
         adanya penyakit jantung
       - `ca` : Indikator negatif kuat. Semakin banyak jumlah pembuluh darah (ca) yang terdeteksi, korelasi dengan penyakit 
         jantung semakin rendah
       - `thal` : Indikator negatif kuat. Nilai hasil tes thal yang lebih tinggi berkorelasi dengan tidak adanya penyakit jantung
       - `sex` : Indikator negatif sedang
       - `age` : Indekator negatif sedang
       - `trestbps`, `chol`, `fbs` : Ketiganya adalah indikator negatif yang sangat lemah
    ''')  

elif nav == 'Data Preprocessing':
    st.title("Data Preprocessing")
    st.write('''
       
    Halo Sobat, kita sudah melakukan kegiatan EDA dan kalian juga sudah mendapatkan insight dari data yang kita miliki.
    Sekarang kita masuk pada tahap preprocessing agar datanya menjadi semakin clean dan rapi, Sobat. Pada tahap ini kita
    akan melakukan beberapa hal yang paling sering dilakukan dalam persiapan pemodelan Machine Learning, antara lain penanganan
    missing values, penanganan data duplikat, data cleaning, penanganan outlier, dan memisahkan kolom numerik dan kategori.
    Ayo kita mulai satu per satu, Sobat!
    ''')  

    st.write(''' 
    **1. Penanganan Missing Values**  
    Sebelumnya kita sudah lihat pada EDA bahwa semua kolom terisi yang terlihat dari adanya keterangan `non-null` atau jumlah `1025`
    pada tiap kolom yang berarti semua kolom terisi. Dengan demikian sampai pada tahap ini masih belum ditemukan missing values. Namun,
    bukan berarti missing values tidak akan muncul lagi, missing values bisa saja muncul sebagai akibat dari hasil penanganan
    data pada tahap atau kegiatan lain.
    ''')

    st.write(''' 
    **2. Penanganan Data Duplikat**  
    Sobat, selanjutnya kita coba cek apakah ada duplikasi data pada dataset ini. Duplikasi data ini bisa terjadi karena kesalahan
    dalam proses pengumpulan data atau kesalahan teknis lainnya. Sobat bisa memeriksa duplikasi data dengan menggunakan fungsi `duplicated()`.
    Jika ada duplikasi data, kita bisa menghapusnya dengan fungsi `drop_duplicates()`. Berikut contoh scriptnya:
    ''')  
    col_check, col_drop = st.columns(2)
    with col_check:
        st.write('''
        **Cek Duplikasi Data**
        ''')
        st.code('''
        # Cek duplikasi data
        df.duplicated().sum()
        ''')  
        st.write('''
        **Hasil Cek Duplikasi Data**  
        Berikut hasil cek duplikasi data pada dataset ini:
        ''')
        st.code(df.duplicated().sum())
        
    with col_drop:
        st.write('''
        **Skrip Drop Duplikasi Data**
        ''')
        st.code('''
        # Menghapus data duplikat
        df.drop_duplicates(inplace=True)
        ''')
        # Hapus duplikasi data
        df.drop_duplicates(inplace=True)
        st.write('''
        **Hasil Cek Duplikasi Data Setelah Dihapus**  
        Berikut hasil cek duplikasi data pada dataset ini setelah dihapus:
        ''')
        st.code(df.duplicated().sum())  
    
    st.write(''' 
    **3. Data Cleaning**  
    Pembersihan data (data cleaning) ini penting banget Sobat, karena data yang kotor atau nggak valid bisa bikin model machine learning
    yang kita buat jadi nggak akurat. Data yang kotor ini bisa berupa data yang nggak valid atau data yang nggak konsisten.
    Pada project ini, kita akan fokus pembersihan data pada data yang nggak valid ya, Sobat.   
    
    Gimana ya biar kita tahu ada data yang tidak valid, Sobat? Ya, betul banget, kita harus paham dulu deskripsi dari tiap kolom/fitur
    pada dataset ini. Dari penjelasan deskripsi dataset di menu Data Collecting. Untuk memudahkan proses belajar, kita bisa fokus cek
    data tidak valid dari semua kolom kategorikal saja ya, Sobat. Terlebih dahulu kita coba tampilkan berapa jumlah nilai unik dari tiap
    kolom kategorikal. Yuk kita lihat datanya di bawah ini:
    ''')

    st.dataframe(df.nunique().to_frame().transpose())

    st.write(''' 
    Nah, dari tabel di atas kita bisa lihat bahwa ada 2 kolom kategorikal yang memiliki nilai unik melebihi deskripsi, yaitu `ca` dan `thal`.
    Berikut penjelasannya, Sobat:
    1. Feature `CA`: Seharusnya hanya memiliki 4 nilai dari rentang 0-3, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
    2. Feature `thal`: Seharusnya hanya memiliki 3 nilai dari rentang 1-3, maka dari itu nilai 0 diubah menjadi NaN (karena seharusnya tidak ada)
    ''')  
    views = st.radio("Show Data", ("CA", "Thal"))
    if views == "CA":
        st.write('''
        **Feature CA**
        
        Feature CA memiliki 4 nilai dari rentang 0-3, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        st.dataframe(df.ca.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**  
        Inilah data `ca` setelah nilai 4 diubah menjadi NaN.
        ''')
        st.dataframe(df.ca.replace(4, np.nan).value_counts().to_frame().transpose())
    elif views == "Thal":
        st.write('''
        **Feature Thal**
        
        Feature Thal memiliki 3 nilai dari rentang 1-3, maka dari itu nilai 0 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        st.dataframe(df.thal.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**  
        Inilah data `thal` setelah nilai 0 diubah menjadi NaN.
        ''')
        st.dataframe(df.thal.replace(0, np.nan).value_counts().to_frame().transpose())  

    st.write('''
    Karena ada nilai yang diubah menjadi NaN, maka sekarang ada missing value pada dataset ini. Oleh karena itu, kita perlu
    melakukan penanganan missing value. Cara yang paling umum dan sederhana yang digunakan untuk menangani missing value
    adalah dengan menghapus semua baris data yang memiliki missing value atau mengisinya dengan nilai `mean`, `median`, atau `mode`.
    Pada project ini, kita akan mengisi missing value dengan nilai `mode` karena kolom yang memiliki missing value adalah kolom
    kategorikal yang memiliki batasan nilai tertentu sesuai deskripsi dataset. Berikut contoh scriptnya:
    ''')  
    col_ca, col_thal = st.columns(2)
    with col_ca:
        st.write('''
        **Script Penanganan Missing Value CA**
        ''')
        st.code('''
        # Fillna pada kolom 'ca' dengan modus
        modus_ca = df['ca'].mode()[0]
        df['ca'] = df['ca'].fillna(modus_ca)
        ''')
        # Fillna pada kolom 'ca' dengan modus
        modus_ca = df['ca'].mode()[0]
        df['ca'] = df['ca'].fillna(modus_ca)
    
    with col_thal:
        st.write('''
        **Script Penanganan Missing Value Thal**
        ''')
        st.code('''
        # Fillna pada kolom 'thal' dengan modus
        modus_thal = df['thal'].mode()[0]
        df['thal'] = df['thal'].fillna(modus_thal)
        ''')  
        # Fillna pada kolom 'thal' dengan modus
        modus_thal = df['thal'].mode()[0]
        df['thal'] = df['thal'].fillna(modus_thal)

        # Ubah tipe data kolom 'ca' dan 'thal' menjadi integer
        df['ca'] = df['ca'].astype(int)
        df['thal'] = df['thal'].astype(int)

    st.write('''
    Berikut hasil setelah penanganan missing value:
    ''')   
    st.dataframe(df.isnull().sum().to_frame().transpose())  

    st.write('''
    **4. Penanganan Outlier**  
    Sobat semua masih ingat kan pada EDA kita sudah melakukan deteksi outlier, di mana outlier ditunjukkan oleh tanda titik-titik
    pada boxplot, lebih tepatnya pada kolom `trestbps`, `chol`, `thalach`, dan `oldpeak` Tapi kita belum tahu kan outliernya 
    sebenarnya ada pada baris data yang mana aja sih. Nah, pada tahap ini kita akan melakukan pengecekan outlier kemudian menanganinya, 
    Sobat. Untuk penanganannya pada project ini kita coba dengan melakukan drop atau hapus data ya, Sobat. Yuk, kita mulai!
    ''')  
    
    script_outliers= '''
    def get_outliers_dataframe(data_input):
    
        # List untuk menampung semua data outlier sebelum dijadikan DataFrame
        outlier_list = []
        
        # Ambil hanya kolom kontinu dari DataFrame input
        data_out = data_input[continous_features]
        
        for each_feature in data_out.columns:
            feature_data = data_out[each_feature]
            Q1 = np.percentile(feature_data, 25.)
            Q3 = np.percentile(feature_data, 75.)
            IQR = Q3 - Q1
            outlier_step = IQR * 1.5
            
            # Temukan baris yang DI LUAR batas IQR
            outliers_mask = ~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))
            
            # Ambil indeks dan nilai dari outlier
            outlier_indices = feature_data[outliers_mask].index
            outlier_values = feature_data[outliers_mask].values
            
            # Tambahkan ke daftar (list) kita
            for index, value in zip(outlier_indices, outlier_values):
                outlier_list.append([index, each_feature, value])

        # Setelah loop selesai, buat satu DataFrame besar dari list tersebut
        if outlier_list:
            outlier_df = pd.DataFrame(outlier_list, columns=['No_Index', 'Fitur', 'Value'])
            return outlier_df
        else:
            return pd.DataFrame(columns=['No_Index', 'Fitur', 'Value']) # Kembalikan DataFrame kosong
    ''' 
    col_scriptout, col_outliers, col_outliersum  = st.columns([6, 3, 2])
    with col_scriptout:
        st.write('''
        Script mencari nilai outlier :  
        ''')
        st.code(script_outliers)

    continous_features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    def get_outliers_dataframe(data_input):

        # List untuk menampung semua data outlier sebelum dijadikan DataFrame
        outlier_list = []
        
        # Ambil hanya kolom kontinu dari DataFrame input
        data_out = data_input[continous_features]
        
        for each_feature in data_out.columns:
            feature_data = data_out[each_feature]
            Q1 = np.percentile(feature_data, 25.)
            Q3 = np.percentile(feature_data, 75.)
            IQR = Q3 - Q1
            outlier_step = IQR * 1.5
            
            # Temukan baris yang DI LUAR batas IQR
            outliers_mask = ~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))
            
            # Ambil indeks dan nilai dari outlier
            outlier_indices = feature_data[outliers_mask].index
            outlier_values = feature_data[outliers_mask].values
            
            # Tambahkan ke daftar (list) kita
            for index, value in zip(outlier_indices, outlier_values):
                outlier_list.append([index, each_feature, value])

        # Setelah loop selesai, buat satu DataFrame besar dari list tersebut
        if outlier_list:
            outlier_df = pd.DataFrame(outlier_list, columns=['No_Index', 'Fitur', 'Value'])
            return outlier_df
        else:
            return pd.DataFrame(columns=['No_Index', 'Fitur', 'Value']) # Kembalikan DataFrame kosong
    
    # 1. Panggil fungsi Anda untuk mendapatkan DataFrame outlier
    outliers_df = get_outliers_dataframe(df) # Berikan DataFrame lengkap 'df'
    
    # 2. Tampilkan hasilnya di Streamlit
    if not outliers_df.empty:
        with col_outliers:
            st.write(f"Ditemukan total {len(outliers_df)} nilai outlier:")
    
            # Tampilkan DataFrame interaktif
            st.dataframe(outliers_df, use_container_width=True, height=735)
        
        with col_outliersum:

            # (Opsional) Tampilkan juga ringkasan per fitur
            st.write('Ringkasan per Fitur')
            st.dataframe(outliers_df.groupby('Fitur').size().reset_index(name='Jumlah'), use_container_width=True)
                
    else:
        st.success("🎉 Selamat! Tidak ditemukan outlier pada fitur kontinu.")

    st.write('''
    Selanjutnya kita akan tangani outlier ini dengan cara yang paling sederhana yaitu drop/hapus. Drop outlier dapat dilakukan dengan
    menggunakan perintah :
    ''')  
    st.code('''
    indices_to_drop = outliers_df['No_Index'].unique()
    df_cleaned = df.drop(index=indices_to_drop)
    '''
    )
    
    indices_to_drop = outliers_df['No_Index'].unique()
    df_cleaned = df.drop(index=indices_to_drop)

    col_rowawal, col_rowdeleted, col_rownow = st.columns(3)
    with col_rowawal:
        st.write("Jumlah baris awal")
        st.code(len(df))
    
    with col_rowdeleted:
        st.write("Jumlah baris yang dihapus")
        st.code(len(indices_to_drop))
    
    with col_rownow:
        st.write(f"Jumlah baris sekarang")
        st.code(len(df_cleaned))  
    
    st.write('''
    Sobat, kita sudah berhasil menghapus outlier dengan cara yang sederhana yaitu membuat variabel baru `df_cleaned` dan
    mengisinya sama persis dengan isi `df` yang sudah dikurangi dengan outlier. Cara ini membuat `df` asli tidak terpengaruh
    oleh penghapusan outlier dan dapat kita gunakan kembali jika perlu memakai data asli.  
    Seperti yang kita lihat, data yang tersisa sejumlah `283` baris. Jumlah ini masih aman untuk model Machine Learning, namun tidak
    disarankan untuk model Neural Networks yang haus data. Sehingga menjadi pilihan kita apakah akan menggunakan
    sisa data ini atau tetap menggunakan data yang belum dihapus outliernya. Sobat semua bisa tentukan sendiri preferensinya ya.
    Pada project ini saya akan menggunakan sisa data yang tersisa untuk dilakukan pemodelan menggunakan Machine Learning
    saja tanpa mencoba model Neural Networks.
    ''')  

    st.write('''
    **5. Memisahkan Kolom Numerik dan Kategorikal**  
    Tahap ini akhir dari tahap data preprocessing kita pada project ini ya, Sobat. Pada tahap ini kita akan memisahkan kolom numerik dengan
    kolom kategorikal. Hal ini berguna agar memudahkan nanti saat melakukan Feature Engineering mengubah tipe numerik ke object. Pemisahan
    kolom ini sederhana saja ya, Sobat. Kita cukup membuat variabel baru yang menampung nama kolom yang bersifat kategori. Nah, berarti
    kita harus tahu dulu mana yang numerik dan mana yang kategori. Sobat bisa melihat ke menu Data Collecting dan EDA lagi untuk tahu mana
    aja nih yang masuk numerik dan kategorikal karena sebenarnya secara tersembunyi saya sudah memisahkan itu di EDA, hehehe. Berikut contoh
    scriptnya: 
    ''')  
    st.code('''
    lst = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    ''')  

elif nav == 'Feature Engineering':
    st.title("Feature Engineering")  
    st.write('''
    Kita lanjut ya, Sobat. Sekarang kita akan masuk ke tahap Feature Engineering. Pada tahap ini, kita akan melakukan 
    beberapa hal seperti mengubah tipe data, melakukan scaling, dimensionality reduction, dan transform data baru untuk
    kebutuhan pemodelan. Hal ini semua dilakukan agar data lebih berkualitas dan sesuai dengan kebutuhan model machine 
    learning yang akan kita buat nanti. Yuk, kita bahas satu per satu!
    ''')
    st.write('''
    **1. Mengubah kolom numerik menjadi object**  
    Pada bagian EDA, kita sudah melihat deskripsi statistik dari tiap kolom. Sobat bisa lihat bahwa ada beberapa kolom 
    yang sebenarnya bersifat kategorikal, tapi pada deskripsi statistik muncul sebagai numerik. Oleh karena itu, kita 
    perlu mengubah tipe data beberapa kolom menjadi `object` agar lebih sesuai dengan sifat aslinya.  
    Pada Data Preprocessing, kita sudah memisahkan kolom kategorikal dengan membuat varibel yang menampung kolom
    kategorikal. Kali ini kita akan manfaatkan variabel itu untuk mengubahnya menjadi `object`, yuk lihat script dan hasilnya!
    ''')  
    lst = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    df_numeric = df.drop(columns=lst)
    col_scriptobj, col_hasil = st.columns(2)
    with col_scriptobj:
        st.write('''
        Script Perubahan Numerik ke Object
        ''')
        #lst = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
        #df[lst] = df[lst].astype(object)
        script_obj = '''
        lst = ['sex', 'cp', 'fbs', 'restecg', 'exang', 
              'slope', 'ca', 'thal', 'target']
        df[lst] = df[lst].astype(object)
        '''
        st.code(script_obj)
        
    with col_hasil:
        st.write('''
        Hasil perubahan
        ''')
        
        df[lst] = df[lst].astype(object)
        # Tangkap output df.info() pasca perubahan
        buffer_new = io.StringIO()
        df[lst].info(buf=buffer_new)
        info_new_string = buffer_new.getvalue()
    
        # Tampilkan di kolom kiri
        st.code(info_new_string)  

    st.write('''
    **2. Scaling Data**  
    Scaling adalah suatu cara untuk membuat numerical data pada dataset memiliki rentang nilai (scale) yang sama, 
    sehingga tidak ada lagi suatu variabel data yang mendominasi variabel data lainnya. Scaling sendiri umumnya ada
    2 yaitu Standard Scaler dan MinMax Scaler. Pada project ini kita coba yang Standard Scaler ya, Sob.
    ''')  
    col_scriptscale, col_hasilscale = st.columns(2)
    with col_scriptscale:
        st.write('''
        Script Scaling
        ''')
        #st.code(info_new_string)
        st.code('''
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        ''')
    
    with col_hasilscale:
        # Mendefinisikan standard scaler
        scaler = StandardScaler()
        df_num = df.drop(columns=lst)
        df_num = scaler.fit_transform(df_num)
        df_num = pd.DataFrame(df_num, columns=df.drop(columns=lst).columns)
        df_lst_reset = df[lst].reset_index(drop=True)
        df_combined = pd.concat([df_num, df_lst_reset], axis=1)
        st.write('''
        Hasil Scaling
        ''')
        st.dataframe(df_combined.head())  
        output_path = 'dataset_scaled.csv'
        df_combined.to_csv(output_path, index=False)

    st.write('''
    **3. Dimensionality Reduction**  
    Sobat, seperti halnya kita mengajarkan sesuatu kepada seseorang, tentunya kita akan melakukan pemilihan hal apa
    yang paling penting untuk dipelajari, harapannya orang tersebut dapat dengan mudah memahami sesuatu tersebut tanpa
    kehilangan informasi penting lainnya. Dimensionality Reduction ini juga serupa dengan hal ini, pada tahap ini 
    akan dilakukan penyederhaan data dengan membatasi pada fitur tertentu yang paling berpengaruh saja tanpa mengurangi 
    informasi penting yang ada dengan harapan kinerja model akan semakin baik. Kegiatan ini dapat dilakukan dengan 
    menggunakan PCA (Principal Component Analysis). Mari kita mulai! 
    ''')  
    script_PCA1 = '''
    # Perform PCA for dimensionality reduction
    from sklearn.decomposition import PCA

    feature_number = df_combined.shape[1]
    pca = PCA(n_components=feature_number)

    # Fit PCA with dataset
    pca.fit(df_combined)

    # Get variance information
    variance_ratio = pca.explained_variance_ratio_

    # Calculate cummulative
    cumulative_variance = np.cumsum(variance_ratio)

    pca = PCA(n_components=9)
    heart_data_reduced = pca.fit_transform(df_combined.drop('target', axis=1))
    '''
    col_scriptPCA1, col_scree = st.columns(2)
    with col_scriptPCA1:
        st.write("Script PCA")
        st.code(script_PCA1)
        # Perform PCA for dimensionality reduction
        from sklearn.decomposition import PCA

        feature_number = df_combined.shape[1]
        pca = PCA(n_components=feature_number)

        # Fit PCA with dataset
        pca.fit(df_combined)

        # Get variance information
        variance_ratio = pca.explained_variance_ratio_

        # Calculate cummulative
        cumulative_variance = np.cumsum(variance_ratio)

        pca = PCA(n_components=9)
        heart_data_reduced = pca.fit_transform(df_combined.drop('target', axis=1))
    with col_scree:
        plt.plot(range(1, len(variance_ratio) + 1), variance_ratio, marker='o')
        plt.xlabel('Komponen Utama ke-')
        plt.ylabel('Varians (Nilai Eigen)')
        plt.title('Scree Plot')
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()

    st.write(''' 
    Berdasarkan Scree Plot di atas, kita bisa lihat bahwa mulai dari titik ke 9 sampai seterusnya sudah
    semakin mendatar yang berarti semakin sedikit informasi baru yang diberikan. Dengan demikian komponen
    ke-10 dan seterusnya dianggap bisa diabaikan dan terpilih hanya 9 komponen saja. Selanjutnya kita akan
    cari tahu ke-9 komponen tersebut. Check this out, Sob!
    ''')  
    script_PCA2 = '''
    pca = PCA(n_components=9)
    heart_data_reduced = pca.fit_transform(df_combined.drop('target', axis=1))

    feature_names = df_combined.drop('target', axis=1).columns.to_list()
    component_names = [f"PC{i+1}" for i in range(pca.n_components_)]

    for component, component_name in zip(pca.components_, component_names):
        feature_indices = np.abs(component).argsort()[::-1] # Sort by absolute value
        retained_features = [feature_names[idx] for idx in feature_indices[:pca.n_components_]] # Get top features up to number of components
        print(f"{component_name}: {retained_features}")
    '''
    col_scriptPCA2, col_featuresPCA = st.columns(2)
    with col_scriptPCA2:
        st.write("Script Cek Kombinasi PCA")
        st.code(script_PCA2)
    
    with col_featuresPCA:
        st.write("Daftar 9 komponen PC")  
        pca = PCA(n_components=9)
        heart_data_reduced = pca.fit_transform(df_combined.drop('target', axis=1))

        feature_names = df_combined.drop('target', axis=1).columns.to_list()
        component_names = [f"PC{i+1}" for i in range(pca.n_components_)]

        results_list = []

        for component, component_name in zip(pca.components_, component_names):
            feature_indices = np.abs(component).argsort()[::-1] # Sort by absolute value
            retained_features = [feature_names[idx] for idx in feature_indices[:pca.n_components_]] # Get top features up to number of components
            results_list.append(f"{component_name}: {retained_features}")
        final_output_string = "\n".join(results_list)
        st.code(final_output_string, language='text')  
    
    st.write(''' 
    Setelah kita mendapatkan hasil PCA, data kita masih berupa array NumPy mentah. Maka, langkah selanjutnya 
    adalah menyiapkan data baru ini agar siap digunakan untuk training model dengan mengkspornya menjadi dataset baru.
    Berikut contoh scriptnya.
    ''')   
    script_exportPCA = '''
    # Mengubah hasil PCA menjadi DataFrame
    heart_data_reduced_df = pd.DataFrame(heart_data_reduced, columns=[f'PC{i+1}' for i in range(heart_data_reduced.shape[1])])

    # Menambahkan kolom target kembali
    # Pastikan indeks sesuai antara heart_data_reduced_df dan df
    heart_data_reduced_df['target'] = df['target'].values
    '''
    st.code(script_exportPCA)
    # Mengubah hasil PCA menjadi DataFrame
    heart_data_reduced_df = pd.DataFrame(heart_data_reduced, columns=[f'PC{i+1}' for i in range(heart_data_reduced.shape[1])])

    # Menambahkan kolom target kembali
    # Pastikan indeks sesuai antara heart_data_reduced_df dan df
    heart_data_reduced_df['target'] = df['target'].values
    output_path = 'dataset_pca.csv'
    heart_data_reduced_df.to_csv(output_path, index=False)

elif nav == 'Modelling':
    st.title("Modelling")  
    st.write('''
    Halo Sobat, akhirnya kita sampai pada tahap Modelling (Pemodelan). Pada tahap ini kita akan menggunakan
    data pasca tahap PCA untuk dijadikan dataset baru. Aku sudah mengekspornya tadi secara diam-diam setelah menampilkan
    dan menjalankan scriptnya, hehehe. Pemodelan ini nanti akan menggunakan 4 model Machine Learning, yaitu:  
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - MLP Classifier  
    
    Langsung aja, ayo kita mulai!
    ''')  
    script_train = ''' 
    # Memuat dataset yang sudah rapi ditahapan sebelumnya
    data_pca = "/content/drive/MyDrive/DQLAB MACHINE LEARNING/dataset_pca.csv"
    df = pd.read_csv(data_pca, index_col=0)

    # 1. Pisahkan Fitur (X) dan Target (y)
    # X adalah semua kolom KECUALI 'target'
    X = df.drop('target', axis=1)

    # y HANYA kolom 'target'
    y = df['target']

    # Bagi data, 80% train, 20% test
    # random_state=42 agar hasil splitnya konsisten setiap kali dijalankan
    # stratify=y penting agar proporsi kelas 'target' di data train & test sama
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Definisikan model yang akan kita gunakan
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "MLP Classifier": MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(10, 5)) 
        # max_iter=1000 agar MLP konvergen
        # hidden_layer_sizes=(10, 5) adalah arsitektur sederhana yg cocok utk data kecil
    }

    # 2. List untuk menyimpan hasil
    results = {}

    # 3. Loop untuk melatih dan evaluasi
    for name, model in models.items():
        print(f"\n" + "="*30)
        print(f"MELATIH MODEL: {name}")
        print("="*30)
        
        # Latih model
        model.fit(X_train, y_train)
        
        # Lakukan prediksi
        y_pred = model.predict(X_test)
        
        # Evaluasi akurasi
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    '''
    st.code(script_train)  
    st.write("Setelah dilakukan training menggunakan Google COlab, inilah hasil akurasinya, Sob")  
    accuracy_score = {
            'Logistic Regression': 0.77,
            'Decision Tree': 0.79,
            'Random Forest': 0.88,
            'MLP Classifier': 0.74,
    }
    st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))  
    st.write('''
    Berdasarkan tabel di atas, kita bisa lihat bahwa nilai terbaik pada model Random Forest, dengan demikian
    kita akan ekspor model tersebut untuk digunakan pada menu Prediction. Berikut contoh scriptnya.
    ''')  

    st.code(''' 
    # Tentukan folder untuk menyimpan model
    model_save_path = "models"

    # Buat folder itu jika belum ada (exist_ok=True agar tidak error jika folder sudah ada)
    os.makedirs(model_save_path, exist_ok=True)
    
    
    ''')

    st.write('''  
    Akhirnya kita selesai melakukan pemodelan Sobat. Cukup panjang ya, tapi ini belum seberapa lho. Project ini masih
    sederhana saja, belum dilakukan hyperparameter tuning dan sebagainya. Tidak masalah, setidaknya hasilnya sudah cukup 
    baik dan bisa digunakan prediksi. Selamat mencoba menu Prediction, Sobat! 
    ''')

elif nav == 'Prediction':
    st.header("My Apps")
    heart()

            
