import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataset(filepath='../stroke.csv'):
    """Memuat dataset dari file CSV"""
    print(f"Memuat dataset dari {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset berhasil dimuat. Shape: {df.shape}")
    return df

def exploratory_data_analysis(df, save_plots=False, output_dir='output'):
    """Melakukan Exploratory Data Analysis"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\n1. Informasi Dataset:")
    print(df.info())
    
    print("\n2. Statistik Deskriptif:")
    print(df.describe())
    
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "Tidak ada missing values")
    
    print(f"\n4. Data Duplikat: {df.duplicated().sum()}")
    
    if 'stroke' in df.columns:
        print("\n5. Distribusi Target Variable (Stroke):")
        print(df['stroke'].value_counts())
        print(f"\nPersentase:")
        print(df['stroke'].value_counts(normalize=True) * 100)
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Correlation matrix
        fitur_numerik_real = df.select_dtypes(include=['number']).columns
        if len(fitur_numerik_real) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[fitur_numerik_real].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nCorrelation matrix disimpan ke {output_dir}/correlation_matrix.png")
        
        if 'stroke' in df.columns:
            plt.figure(figsize=(8, 6))
            df['stroke'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
            plt.title('Distribusi Target Variable (Stroke)', fontsize=16)
            plt.xlabel('Stroke (1=Ya, 0=Tidak)', fontsize=12)
            plt.ylabel('Jumlah', fontsize=12)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Target distribution disimpan ke {output_dir}/target_distribution.png")

def detect_outliers_iqr(df, column):
    """Fungsi untuk mendeteksi outlier menggunakan metode IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def data_preprocessing(df, handle_outliers=True, normalize=True, drop_id=True):
    """Melakukan data preprocessing"""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    df_processed = df.copy()
    
    print("\n1. Menghapus baris dengan smoking_status = 'Unknown'...")
    initial_shape = df_processed.shape
    if 'smoking_status' in df_processed.columns:
        # 1. Hapus baris dengan "Unknown" (case-insensitive)
        df_processed = df_processed[~df_processed['smoking_status'].str.lower().isin(['unknown'])]
        removed_count = initial_shape[0] - df_processed.shape[0]
        print(f"   Baris yang dihapus: {removed_count}")
        print(f"   Data sebelum: {initial_shape[0]} baris, setelah: {df_processed.shape[0]} baris")
    else:
        print("   Kolom 'smoking_status' tidak ditemukan.")
    
    # 2. Menghapus baris dengan missing values pada kolom bmi
    print("\n2. Menghapus baris dengan missing values pada kolom 'bmi'...")
    initial_shape = df_processed.shape
    if 'bmi' in df_processed.columns:
        missing_bmi_count = df_processed['bmi'].isnull().sum()
        if missing_bmi_count > 0:
            df_processed = df_processed.dropna(subset=['bmi'])
            removed_count = initial_shape[0] - df_processed.shape[0]
            print(f"   Baris dengan missing bmi: {missing_bmi_count}")
            print(f"   Baris yang dihapus: {removed_count}")
            print(f"   Data sebelum: {initial_shape[0]} baris, setelah: {df_processed.shape[0]} baris")
        else:
            print("   Tidak ada missing values pada kolom 'bmi'.")
    else:
        print("   Kolom 'bmi' tidak ditemukan.")
    
    # 3. Encoding Data Kategorikal (dilakukan setelah cleaning)
    print("\n3. Encoding Data Kategorikal...")
    categorical_features = df_processed.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
        print(f"   - Kolom '{col}' selesai di-encode.")
    
    print("   Encoding selesai.")
    
    # Update fitur_numerik setelah encoding (karena kategorikal sekarang numerik)
    fitur_numerik = df_processed.select_dtypes(include=['number']).columns
    
    # 4. Penanganan Missing Values (Imputation untuk kolom selain bmi)
    print("\n4. Penanganan Missing Values (Imputation)...")
    missing_before = df_processed[fitur_numerik].isnull().sum().sum()
    
    if missing_before > 0:
        print(f"   Total missing values sebelum imputation: {missing_before}")
        print(f"   Missing values per kolom:")
        for col in fitur_numerik:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                print(f"     - {col}: {missing_count} missing values")
        
        fitur_numerik_to_impute = [col for col in fitur_numerik if col not in ['stroke', 'bmi']]
        
        if len(fitur_numerik_to_impute) > 0:
            imputer = SimpleImputer(strategy='median')
            df_processed[fitur_numerik_to_impute] = imputer.fit_transform(df_processed[fitur_numerik_to_impute])
            print(f"   Missing values diisi dengan median untuk kolom: {fitur_numerik_to_impute}")
        
        missing_after = df_processed[fitur_numerik].isnull().sum().sum()
        print(f"   Total missing values setelah imputation: {missing_after}")
    else:
        print("   Tidak ada missing values.")
    
    # 5. Penanganan Outlier menggunakan IQR Method (Capping)
    if handle_outliers:
        print("\n5. Penanganan Outlier (IQR Method dengan Capping)...")
        # Exclude target variable dan id dari outlier handling
        fitur_numerik_for_outlier = [col for col in fitur_numerik if col not in ['stroke', 'id', 'Dataset']]
        outlier_info = {}
        
        for col in fitur_numerik_for_outlier:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            before_count = len(df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)])
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            after_count = len(df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)])
            print(f"   - Kolom '{col}': {before_count} outlier di-cap")
        
        print("   Outlier handling selesai.")
    
    # 6. Drop kolom 'id' jika ada
    if drop_id and 'id' in df_processed.columns:
        print("\n6. Menghapus kolom 'id'...")
        df_processed = df_processed.drop('id', axis=1)
        print("   Kolom 'id' berhasil dihapus.")
    
    # 7. Normalisasi/Standarisasi Fitur menggunakan MinMaxScaler
    if normalize:
        print("\n7. Normalisasi Fitur menggunakan MinMaxScaler...")
        # Pisahkan fitur dan target
        if 'stroke' in df_processed.columns:
            X = df_processed.drop('stroke', axis=1)  # Fitur
            y = df_processed['stroke']  # Target
        else:
            X = df_processed.copy()
            y = None
        
        # Normalisasi fitur numerik
        fitur_numerik_after_encoding = X.select_dtypes(include=['number']).columns
        scaler = MinMaxScaler()
        X_scaled = X.copy()
        X_scaled[fitur_numerik_after_encoding] = scaler.fit_transform(X[fitur_numerik_after_encoding])
        
        print(f"   Fitur yang dinormalisasi: {len(fitur_numerik_after_encoding)} fitur")
        print(f"   Range nilai setelah normalisasi: 0 - 1")
        
        # Gabungkan kembali dengan target
        if y is not None:
            df_processed = X_scaled.copy()
            df_processed['stroke'] = y
        else:
            df_processed = X_scaled.copy()
        
        print("   Normalisasi selesai.")
    
    # Konversi eksplisit semua kolom ke numerik (kecuali target jika perlu)
    print("\n8. Konversi tipe data...")
    for col in df_processed.columns:
        if col != 'stroke':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    # Pastikan stroke adalah integer
    if 'stroke' in df_processed.columns:
        df_processed['stroke'] = df_processed['stroke'].astype(int)
    print("   Konversi tipe data selesai.")
    
    # Validasi distribusi target
    if 'stroke' in df_processed.columns:
        print("\n9. Validasi Distribusi Target...")
        stroke_dist = df_processed['stroke'].value_counts()
        print(f"   Distribusi stroke setelah preprocessing:")
        for val, count in stroke_dist.items():
            print(f"     - Stroke {val}: {count} sampel ({count/len(df_processed)*100:.2f}%)")
        
    
    print("\n" + "="*60)
    print("Data preprocessing selesai!")
    print(f"Shape data setelah preprocessing: {df_processed.shape}")
    print("="*60)
    
    return df_processed, encoders, scaler if normalize else None

def save_preprocessed_data(df, output_path='stroke_preprocessed.csv'):
    """Menyimpan data yang sudah dipreprocessing"""
    print(f"\nMenyimpan data preprocessed ke {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan. Shape: {df.shape}")

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline"""
    print("="*60)
    print("AUTOMATION SCRIPT - PREPROCESSING DATASET STROKE")
    print("="*60)
    
    input_file = '../stroke.csv'
    output_file = 'stroke_preprocessed.csv'
    output_dir = 'output'
    perform_eda = True  
    save_plots = False 
    
    # 1. Load Dataset
    df = load_dataset(input_file)
    
    # 2. Exploratory Data Analysis (opsional)
    if perform_eda:
        exploratory_data_analysis(df, save_plots=save_plots, output_dir=output_dir)
    
    # 3. Data Preprocessing
    df_processed, encoders, scaler = data_preprocessing(
        df, 
        handle_outliers=True, 
        normalize=True, 
        drop_id=True
    )
    
    # 4. Simpan data yang sudah dipreprocessing
    save_preprocessed_data(df_processed, output_path=output_file)
    
    # 5. Tampilkan preview data final
    print("\nPreview data setelah preprocessing:")
    print(df_processed.head())
    print("\nInfo data final:")
    print(df_processed.info())
    
    print("\n" + "="*60)
    print("SELESAI! Pipeline preprocessing berhasil dijalankan.")
    print("="*60)
    
    return df_processed, encoders, scaler

if __name__ == "__main__":
    df_processed, encoders, scaler = main()

