from logging import PlaceHolder
import streamlit as st
import pandas as pd
import pickle
import sklearn

def main():
	dataset = pd.read_csv('DataPreparation.csv')

	st.title('Students Adaptability Level Online Education')
			
	# gender = st.text_input('Masukkan Jenis Kelamin')
	genders = st.radio(
			'Jenis Kelamin',
			('Boy', 'Girl')
	)

	# age = st.text_input('Masukkan Umur')
	ages = (st.text_input('Masukkan Umur', placeholder="Masukkan nilai dengan range 5 tahun sekali, contoh : 10-15"))
	age = str(ages)
	
	# education = st.text_input('Masukkan Education Level')
	educations = st.radio(
			'Education Level',
			('College', 'School', 'University')
	)
			
	# institution = st.text_input('Masukkan Institution Type')
	institutions = st.radio(
			'Institution Type',
			('Non Government', 'Government')
	)

	# ITStudent = st.text_input('Apakah IT Student')
	ITStudents = st.radio(
			'IT Student',
			('Yes', 'No')
	)
	
	# Location = st.text_input('Location')
	Locations = st.radio(
			'Location',
			('Yes', 'No')
	)
	
	# LoadShedding = st.text_input('Load Shedding')
	LoadSheddings = st.radio(
			'Load Shedding',
			('Low', 'High')
	)

	# Financial = st.text_input('Financial Condition')
	Financials = st.radio(
			'Financial Condition',
			('Mid', 'Poor', 'Rich')
	)
	
	# Internet = st.text_input('Internet Type')
	Internets = st.radio(
			'Internet Type',
			('Mobile Data', 'Wifi')
	)
	
	# Network = st.text_input('Network Type')
	Networks = st.radio(
			'Network Type',
			('2G', '3G', '4G')
	)
	
	# Duration = st.text_input('Class Duration')
	Durations = st.radio(
			'Class Duration',
			('1-3', '3-6', '0')
	)
			
	# lms = st.text_input('Self Lms')
	lmss = st.radio(
			'Self Lms',
			('Yes', 'No')
	)

	# Device = st.text_input('Device')
	Devices = st.radio(
			'Device',
			('Mobile', 'Tab', 'Computer')
	)

	new_val = pd.DataFrame([[genders, age, educations, institutions, ITStudents, Locations, LoadSheddings, Financials, Internets, Networks, Durations, lmss, Devices]])

	infile = open('model_SVM.pkl', 'rb')
	svm_model = pickle.load(infile)
	infile.close()

	infile = open('enc.pkl', 'rb')
	encoding = pickle.load(infile)
	infile.close()

	new_val = encoding.transform(new_val)

	y_pred = svm_model.predict(new_val)
 
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

	
if __name__ == '__main__':
	main()
