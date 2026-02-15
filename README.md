# IT5006-Group-Project
!Dashboard Running Instructions!
1. Install dependencies：
pip install -r requirements.txt
2. Navigate to the project directory
3. Generate the processed dataset
Run the preprocessing notebook to generate the cleaned CSV file:

notebooks/preprocessing.ipynb

After running the notebook, make sure the following file exists:

processed/chicago_crimes_2015_2024_cleaned.csv
4. Run the Streamlit dashboard：
streamlit run dashboard/InteractiveDashboard.py
