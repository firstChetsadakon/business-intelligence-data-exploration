
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
import scipy.stats as stats

# ปรับให้ Plot เป็นภาษาไทย
import matplotlib as mpl
mpl.font_manager.fontManager.addfont('./font/Sarabun-Regular.ttf')
mpl.rc('font', family='Sarabun')
# ตั้งค่าฟอนต์แบบ Global
# **ตั้งค่า rcParams เพื่อปรับฟอนต์**
plt.rcParams.update({
    'font.size': 20,         # ขนาดฟอนต์ทั้งหมด
    'axes.titlesize': 21,    # ขนาดฟอนต์ Title ของกราฟ
    'axes.labelsize': 20,    # ขนาดฟอนต์ Label แกน X และ Y
    'xtick.labelsize': 20,   # ขนาดฟอนต์ของ Tick ในแกน X
    'ytick.labelsize': 20,   # ขนาดฟอนต์ของ Tick ในแกน Y
    'legend.fontsize': 20    # ขนาดฟอนต์ของ Legend (ถ้ามี)
})

def plot_missing_value(df):
    # Missing Value
    missing_df = df.isnull().sum()
    missing_df = missing_df[missing_df>0].sort_values(ascending=False)
    plt.figure(figsize = (20,6))
    plt.title('Missing Value Of Each Column')
    ax = missing_df.plot(kind='bar', color='skyblue')

    # เพิ่มตัวเลขบนกราฟ
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}',  # ค่า missing value
                    (p.get_x() + p.get_width() / 2, p.get_height()),  # ตำแหน่ง
                    ha='center', va='bottom', fontsize=18)

    plt.show()


def plot_nunique_values(df, columns):
    """
    พล็อตกราฟแสดงจำนวนค่าไม่ซ้ำ (nunique) ในแต่ละคอลัมน์ที่กำหนด
    
    Parameters:
        df (pd.DataFrame): DataFrame ที่ใช้วิเคราะห์
        columns (list): รายชื่อคอลัมน์ที่ต้องการตรวจสอบ
    """
    nunique_counts = df[columns].nunique().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(nunique_counts.index, nunique_counts.values, color='skyblue')
    plt.xlabel("Feature")
    plt.ylabel("Number of Unique Values")
    plt.title("Unique Value Counts per Feature")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # แสดงตัวเลขบนแท่งกราฟ
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
    plt.show()

def plot_value_counts(df, columns):
    """
    พล็อตกราฟแสดงปริมาณค่าต่างๆ ในแต่ละคอลัมน์ที่กำหนด โดยปรับขนาดของกราฟให้เหมาะสมกับจำนวนค่าที่ไม่ซ้ำกัน
    
    Parameters:
        df (pd.DataFrame): DataFrame ที่ใช้วิเคราะห์
        columns (list): รายชื่อคอลัมน์ที่ต้องการตรวจสอบ
    """
    for col in columns:
        value_counts = df[col].value_counts()
        num_unique = len(value_counts)
        
        fig_width = min(max(num_unique / 2, 10), 20)  # ปรับความกว้างตามจำนวนค่า
        fig_height = min(max(num_unique / 5, 5), 15)  # ปรับความสูงตามจำนวนค่า
        
        plt.figure(figsize=(fig_width, fig_height))
        bars = plt.bar(value_counts.index, value_counts.values, color='skyblue')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Value Counts in {col}")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # แสดงตัวเลขบนแท่งกราฟ
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        
        plt.show()

def plot_distributions(series, title="Distribution Plots"):
    """
    พล็อตกราฟ Histogram, QQ-Plot, Boxplot พร้อมการทดสอบ Kolmogorov-Smirnov Test
    เพื่อดูว่าข้อมูลมีการแจกแจงแบบ Normal Distribution หรือไม่
    
    Parameters:
        series (pd.Series): ชุดข้อมูลที่ต้องการตรวจสอบ
        title (str): ชื่อกราฟหลักที่จะแสดงในส่วนบนของกราฟ (default="Distribution Plots")
    """
    # Kolmogorov-Smirnov test
    # ks_stat, ks_p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
    
    # # ตั้งค่า seaborn theme
    # sns.set_theme(style="whitegrid")
    
    # สร้าง subplot 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram
    sns.histplot(series, kde=True, ax=axes[0])  # ใช้สีดั้งเดิม
    axes[0].set_title('Histogram & KDE')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    
    # 2. QQ-Plot
    stats.probplot(series, dist="norm", plot=axes[1])  # ใช้สีดั้งเดิม
    axes[1].set_title('QQ-Plot')
    
    # 3. Boxplot
    sns.boxplot(x=series, ax=axes[2])  # ใช้สีดั้งเดิม
    axes[2].set_title('Boxplot')
    
    # เพิ่มสรุป Kolmogorov-Smirnov test ใน subplot
    # fig.text(0.5, 0.95, f'{title}\nK-S Test: Stat={ks_stat:.4f}, p-value={ks_p_value:.4f}', 
    #          ha='center', fontsize=20, fontweight='bold')
    fig.suptitle(title)
    # แสดงผล
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def plot_heatmap(df, columns, title="Heatmap of Selected Columns"):
    """
    ฟังก์ชันนี้ใช้สร้าง heatmap โดยเลือกเฉพาะคอลัมน์ที่กำหนด

    Parameters:
        df (pd.DataFrame): DataFrame ที่ใช้สร้าง heatmap
        columns (list): รายชื่อคอลัมน์ที่ต้องการใช้
        title (str): ชื่อกราฟ heatmap (ค่าเริ่มต้นคือ "Heatmap of Selected Columns")

    Returns:
        None (แสดงผล heatmap)
    """
    # ดึงเฉพาะคอลัมน์ที่สนใจ
    selected_df = df[columns]

    # คำนวณ correlation matrix
    corr_matrix = selected_df.corr()

    # สร้าง Heatmap
    plt.figure(figsize=(20, 8))  # ปรับขนาด
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)

    # ตั้งค่าชื่อกราฟ
    plt.title(title, fontsize=20, fontweight='bold')

    # แสดงผล
    plt.show()

def get_high_correlation_pairs(df,selected_column, threshold=0.50):
    """
    ค้นหาคู่คอลัมน์ที่มีค่า correlation มากกว่าค่าที่กำหนด (default=0.50)
    
    Parameters:
        df (pd.DataFrame): DataFrame ที่ใช้คำนวณ correlation
        threshold (float): ค่าที่ใช้เป็นเกณฑ์คัดเลือก correlation (default=0.50)
    
    Returns:
        pd.DataFrame: DataFrame ที่มีคอลัมน์ ['Feature 1', 'Feature 2', 'Correlation']
    """
    # คำนวณ Correlation Matrix
    corr_matrix = df[selected_column].corr()

    # สร้าง DataFrame ของคู่คอลัมน์ที่มีค่า correlation สูง
    high_corr_pairs = (
        corr_matrix.where(lambda x: abs(x) > threshold)  # กรองค่า > threshold
        .stack()  # แปลงเป็น Series คู่คอลัมน์
        .reset_index()  # รีเซ็ต index เพื่อให้เป็น DataFrame
    )

    # เปลี่ยนชื่อคอลัมน์
    high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

    # ลบค่าที่ซ้ำซ้อน (เนื่องจาก matrix มีค่า correlation ซ้ำกัน)
    high_corr_pairs = high_corr_pairs[high_corr_pairs["Feature 1"] != high_corr_pairs["Feature 2"]]

    # จัดเรียงค่า correlation จากมากไปน้อย
    high_corr_pairs = high_corr_pairs.sort_values(by="Correlation", ascending=False)

    return high_corr_pairs.reset_index(drop=True)