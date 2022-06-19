from logging import PlaceHolder
import streamlit as st
import pandas as pd
import pickle

dataset = pd.read_csv('DataPreparation.csv')

st.title('Students Adaptability Level Online Education')

try:
    
    # gender = st.text_input('Masukkan Jenis Kelamin')
    genders = st.radio(
        'Jenis Kelamin',
        ('Laki-Laki', 'Perempuan')
    )
    if genders == 'Boy':
        gender = 'Boy'
    else:
        gender = 'Girl'

    # age = st.text_input('Masukkan Umur')
    ages = (st.text_input('Masukkan Umur'))
    age = str(ages)
    
    # education = st.text_input('Masukkan Education Level')
    educations = st.radio(
        'Education Level',
        ('College', 'School', 'University')
    )
        
    # institution = st.text_input('Masukkan Institution Type')
    institutions = st.radio(
        'Institution Type',
        ('Bukan Pemerintah', 'Pemerintah')
    )
    if institutions == 'Bukan Pemerintah':
        institution = 'Non Government'
    else:
        institution = 'Government' 

    # ITStudent = st.text_input('Apakah IT Student')
    ITStudent = 0
    ITStudents = st.radio(
        'IT Student',
        ('Yes', 'No')
    )
    
    # Location = st.text_input('Location')
    Location = 0
    Locations = st.radio(
        'Location',
        ('Yes', 'No')
    )
    
    # LoadShedding = st.text_input('Load Shedding')
    LoadShedding = 0
    LoadSheddings = st.radio(
        'Load Shedding',
        ('Low', 'High')
    )
   
    # Financial = st.text_input('Financial Condition')
    Financial = 0
    Financials = st.radio(
        'Financial Condition',
        ('Mid', 'Poor', 'Rich')
    )
    
    # Internet = st.text_input('Internet Type')
    Internet = 0
    Internets = st.radio(
        'Internet Type',
        ('Mobile Data', 'Wifi')
    )
    
    # Network = st.text_input('Network Type')
    Network = 0
    Networks = st.radio(
        'Network Type',
        ('2G', '3G', '4G')
    )
    
    # Duration = st.text_input('Class Duration')
    Duration = 0
    Durations = st.radio(
        'Class Duration',
        ('1-3', '3-6', '0')
    )
        
    # lms = st.text_input('Self Lms')
    lms = 0
    lmss = st.radio(
        'Self Lms',
        ('Yes', 'No')
    )
   
    # Device = st.text_input('Device')
    Device = 0
    Devices = st.radio(
        'Device',
        ('Mobile', 'Tab', 'Computer')
    )

    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        text-align: center;
        color: #270;
        background-color: #DFF2BF;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="big-font">Hasil Klasifikasi : {y_pred}</p>', unsafe_allow_html=True)
    
    new_val = pd.DataFrame([[gender, age, education, institution, ITStudent, Location, LoadShedding, Financial, Internet, Network, Duration, lms, Device]])

    infile = open('model_SVM.pkl', 'rb')
    svm_model = pickle.load(infile)
    infile.close()

    infile = open('enc.pkl', 'rb')
    encoding = pickle.load(infile)
    infile.close()

    new_val = encoding.transform(new_val)


    y_pred = svm_model.predict(new_val)
    
except:
    pass
