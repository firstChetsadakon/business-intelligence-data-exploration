import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
import scipy.stats as stats
import json
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.preprocessing import OneHotEncoder


# ------------- จัดกลุ่ม Feature ------------------------
def group_root_cause(cause):
    cause = str(cause)
    
    if any(x in cause for x in [ 'ขับรถเร็ว', 'ย้อนศร', 'ขับรถตามกระชั้นชิด', 'แซงรถ', 'เปลี่ยนช่องทาง', 
        'ขับรถผิดช่องทาง', 'คร่อมเส้น', 'ช้าเกินเนื่องจากลักษณะของยานพาหนะ' ]):
        return 'พฤติกรรมผู้ขับขี่'
    elif any(x in cause for x in ['เมาสุรา', 'มึนเมา', 'หลับใน', 'โรคประจำตัว', 'อาการป่วย', 'ไร้ความสามารถทางประสาท']):
        return 'ความไม่พร้อมของผู้ขับขี่'
    elif any(x in cause for x in ['ฝ่าฝืนสัญญาณไฟ', 'ฝ่าฝืนป้ายหยุด', 
        'ไม่ยอมให้รถที่มีสิทธิ์ไปก่อน', 'ไม่ให้สัญญาณชะลอ', 'ไม่ให้สัญญาณเข้าจอด', 'ไม่ให้สัญญาณออกจาก' ]):
        return 'การไม่เคารพกฎจราจร'
    elif any(x in cause for x in ['ตัดหน้า', 'หยุดรถกะทันหัน', 'ความประมาท', 'วิ่งตัดหน้า','สูญเสียการควบคุม']):
        return 'สิ่งตัดหน้า/คาดไม่ถึง'
    elif any(x in cause for x in ['ถนนลื่น', 'ถนนชำรุด', 'ทางโค้งอันตราย', 'แสงสว่างไม่เพียงพอ','ระยะปลอดภัยข้างทางไม่เพียงพอ', 'ระยะการมองเห็นไม่เพียงพอ',
        'ป้ายจราจรถูกบดบัง', 'ทำงานบนถนน']):
        return 'ถนนและสิ่งแวดล้อม'
    elif any(x in cause for x in ['ยางเสื่อมสภาพ', 'ยางแตก', 'ระบบห้ามล้อ', 'เบรกชำรุด','ระบบบังคับเลี้ยวขัดข้อง', 'อุปกรณ์ยานพาหนะบกพร่อง',
        'ระบบไฟฟ้าของยานพาหนะขัดข้อง', 'ระบบสัญญาณไฟจราจรขัดข้อง','รถเสียไม่แสดงเครื่องหมาย']):
        return 'อุปกรณ์/ระบบยานพาหนะ'
    elif any(x in cause for x in ['โทรศัพท์', 'สิ่งรบกวนภายในรถ', 'สิ่งรบกวนภายนอกรถ','การกระทำที่สุ่มเสี่ยง']):
        return 'สิ่งรบกวน/สมาธิ'
    elif any(x in cause for x in ['ไม่คุ้นเคยเส้นทาง', 'ขับรถไม่ชำนาญ']):
        return 'ไม่ชำนาญ/ไม่รู้เส้นทาง'
    else:
        return 'อื่นๆ'
    
def group_accident_type(x):
    x = str(x)
    if any(word in x for word in ['ชนท้าย', 'ชนด้านข้าง', 'ชนในทิศทางตรงกันข้าม', 'ชนเป็นมุม', 'เลี้ยว/ถอยชน']):
        return 'ชนกับยานพาหนะ'
    elif any(word in x for word in ['ชนคนเดินเท้า', 'ชนสิ่งกีดขวาง']):
        return 'ชนกับคนหรือสิ่งของ'
    elif 'พลิกคว่ำ' in x or 'ตกถนน' in x:
        return 'พลิกคว่ำ/ตกถนน'
    else:
        return 'อื่นๆ'
    
def group_weather(w):
    w = str(w)
    if 'แจ่มใส' in w:
        return 'อากาศแจ่มใส'
    elif any(x in w for x in ['ฝน','มืดครึ้ม', 'หมอก', 'ควัน', 'ฝุ่น']):
        return 'วิสัยทัศน์ไม่ดี'
    elif any(x in w for x in ['ภัยธรรมชาติ', 'น้ำท่วม', 'พายุ']):
        return 'ภัยธรรมชาติ'
    else:
        return 'อื่นๆ'
    
def categorize_vehicle(Car):
    if pd.isna(Car):
        return 'อื่นๆ'
    
    if Car in ['รถจักรยานยนต์', 'รถจักรยาน', 'รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ','รถยนต์นั่งส่วนบุคคล',
             'รถปิคอัพบรรทุก 4 ล้อ','รถปิคอัพบรรทุก4ล้อ', 'รถปิคอัพโดยสาร', 'รถตู้', 'รถสามล้อ', 'รถสามล้อเครื่อง']:
        return 'รถขนาดเล็ก/ส่วนบุคคล'
    
    elif Car in ['รถบรรทุก 6 ล้อ', 'รถบรรทุกมากกว่า 6 ล้อ ไม่เกิน 10 ล้อ',
                 'รถบรรทุก6ล้อ', 'รถบรรทุกไม่เกิน10ล้อ','รถบรรทุกมากกว่า10ล้อ'
               'รถบรรทุกมากกว่า 10 ล้อ (รถพ่วง)', 'รถอีแต๋น/เพื่อการเกษตร','รถอีแต๋น']:
        return 'รถบรรทุก/ขนส่งสินค้า'
    
    elif Car in ['รถโดยสารขนาดใหญ่','รถโดยสารมากกว่า4ล้อ']:
        return 'รถโดยสารสาธารณะ'
    
    elif Car == 'คนเดินเท้า':
        return 'คนเดินเท้า'
    elif Car == 'อื่นๆ':
        return 'อื่นๆ'
    else:
        return 'อื่นๆ'
    
def categorize_location(text):
    # เช็คว่าข้อมูลเป็น NaN หรือ None
    if pd.isna(text):
        return text  # Return ตัวมันเองในกรณีที่ missing value
    elif 'ทางตรง' in text:
        return 'ทางตรง'
    elif 'ทางโค้ง' in text:
        return 'ทางโค้ง'
    elif 'ทางเชื่อม' in text:
        return 'ทางเชื่อม'
    elif 'แยก' in text:
        return 'ทางแยก'
    else:
        return 'อื่น ๆ'
    

def group_categorical_feature(df):
    df['กลุ่มบริเวณที่เกิดเหตุ'] = df['บริเวณที่เกิดเหตุ'].apply(categorize_location)
    df['กลุ่มลักษณะการเกิดเหตุ'] = df['ลักษณะการเกิดอุบัติเ'].apply(group_accident_type)
    df['กลุ่มมูลเหตุ'] = df['มูลเหตุสันนิษฐาน'].apply(group_root_cause)
    df['กลุ่มสภาพอากาศ'] = df['สภาพอากาศ'].apply(group_weather)
    df['vehicle_group'] = df['รถคันที่1'].apply(categorize_vehicle)
    return df



# ---------------- Fill Missing Value ---------------
def fill_missing_with_mode(df, columns):
    """
    เติมค่า Missing Value ของคอลัมน์ที่กำหนดด้วยค่า mode

    Parameters:
        df (pd.DataFrame): DataFrame ที่ต้องการแก้ไข
        columns (list): รายชื่อคอลัมน์ที่ต้องการเติมค่า missing

    Returns:
        pd.DataFrame: DataFrame ที่เติมค่า missing แล้ว
    """
    df = df.copy()  # เพื่อไม่ให้แก้ไข DataFrame ต้นฉบับโดยตรง
    for col in columns:
        if col in df.columns:
            # คำนวณค่า mode
            mode_value = df[col].mode().iloc[0]

            # นับค่าก่อนเติม
            value_counts_before = df[col].value_counts(dropna=False)

            # เติมค่า missing ด้วย mode
            df[col].fillna(mode_value, inplace=True)
    return df



def fill_missing_with_median(df, columns):
    """
    เติมค่า Missing Value ของคอลัมน์ที่กำหนดด้วยค่า median ของแต่ละคอลัมน์

    Parameters:
        df (pd.DataFrame): DataFrame ที่ต้องการแก้ไข
        columns (list): รายชื่อคอลัมน์ที่ต้องการเติมค่า missing

    Returns:
        pd.DataFrame: DataFrame ที่เติมค่า missing แล้ว
    """
    df = df.copy()  # ป้องกันการแก้ไข DataFrame ต้นฉบับ
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):  # ตรวจสอบว่าเป็นคอลัมน์ตัวเลข
            median_value = df[col].median()  # หา median ของคอลัมน์
            df[col].fillna(median_value, inplace=True)  # เติมค่า NaN ด้วย median
            df[col] = df[col].apply(np.floor)
    return df

def fill_missing_vehicle(row, priority_list):
    if pd.isna(row['รถคันที่1']):  # ตรวจสอบว่าค่าเป็น NaN หรือไม่
        for vehicle in priority_list:
            if vehicle in row and row[vehicle] > 0:  # ตรวจสอบว่ามีรถประเภทนั้นหรือไม่
                return vehicle  # คืนค่าประเภทรถที่พบเป็นอันดับแรก
    return row['รถคันที่1']  # ถ้าไม่ใช่ NaN ให้คืนค่าของเดิม


# ฟังก์ชันแมปจังหวัดเป็นภาค
def get_region(province):

    # สำหรับ Map ข้อมูล ภาคและจังหวัด
    region_mapping = {
        'เหนือ': ['เชียงใหม่', 'เชียงราย','ตาก', 'ลำปาง', 'ลำพูน', 'แพร่', 'น่าน', 'พะเยา', 'แม่ฮ่องสอน', 'อุตรดิตถ์', 'สุโขทัย'],
        'ใต้': ['กระบี่', 'ชุมพร', 'ตรัง', 'นครศรีธรรมราช', 'นราธิวาส', 'ปัตตานี', 'พังงา', 'พัทลุง', 'ภูเก็ต', 'ระนอง', 'สงขลา', 'สตูล', 'สุราษฎร์ธานี', 'ยะลา'],
        'ตะวันออก': ['จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ตราด', 'ปราจีนบุรี', 'ระยอง', 'สระแก้ว'],
        'ตะวันตก': ['กาญจนบุรี', 'ประจวบคีรีขันธ์', 'เพชรบุรี', 'ราชบุรี'],
        'ตะวันออกเฉียงเหนือ': ['กาฬสินธุ์', 'ขอนแก่น', 'ชัยภูมิ', 'นครพนม', 'นครราชสีมา', 'บึงกาฬ', 'บุรีรัมย์', 'มหาสารคาม', 
                            'มุกดาหาร', 'ยโสธร', 'ร้อยเอ็ด', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สุรินทร์', 'หนองคาย', 'หนองบัวลำภู', 
                            'อำนาจเจริญ', 'อุดรธานี', 'อุบลราชธานี'],
        'กลาง': ['กำแพงเพชร', 'ชัยนาท', 'นครนายก', 'นครปฐม', 'นนทบุรี', 'ปทุมธานี', 'อยุธยา', 'พิษณุโลก', 
                'พิจิตร', 'เพชรบูรณ์', 'ลพบุรี', 'สมุทรปราการ', 'สมุทรสาคร', 'สมุทรสงคราม', 'สระบุรี', 'สิงห์บุรี', 'สุพรรณบุรี', 
                'อ่างทอง', 'อุทัยธานี','นครสวรรค์'],
        'กรุงเทพ': ['กรุงเทพมหานคร']
    }

    if pd.isna(province):  # ตรวจสอบ NaN
        return np.nan
    for region, provinces in region_mapping.items():
        if province in provinces:
            return region
    return np.nan  # ถ้าไม่พบ ให้เป็น NaN

def recalculate_accident_number(df):
    # ปรับความ Make Sense ของข้อมูล
    car_column = ['รถจักรยานยนต์', 'รถสามล้อเครื่อง', 'รถยนต์นั่งส่วนบุคคล', 'รถตู้',
        'รถปิคอัพโดยสาร', 'รถโดยสารมากกว่า4ล้อ', 'รถปิคอัพบรรทุก4ล้อ',
        'รถบรรทุก6ล้อ', 'รถบรรทุกไม่เกิน10ล้อ', 'รถบรรทุกมากกว่า10ล้อ',
        'รถอีแต๋น', 'อื่นๆ']


    df['จำนวนรถที่เกิดเหตุ'] = df[car_column].sum(axis=1)
    df['รวมจำนวนผู้บาดเจ็บ'] = df['จำนวนผู้บาดเจ็บเล็ก'] + df['จำนวนผู้บาดเจ็บสาหัส']
    df['จำนวนที่เกิดเหตุทั้ง'] = df[car_column + ['คนเดินเท้า']].sum(axis=1)
    df['รวมผู้ได้รับผลกระทบ'] = df[['จำนวนผู้เสียชีวิต', 'จำนวนผู้บาดเจ็บสาหัส', 'จำนวนผู้บาดเจ็บเล็ก']].sum(axis=1) 
    return df


# ---------------- Deal with outlier --------------------------
def cap_outlier(df,numerical_feature):
    for columns in numerical_feature:
        if pd.api.types.is_numeric_dtype(df[columns]):
            Q1 = df[columns].quantile(0.25)
            Q3 = df[columns].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # แทนค่าที่เกิน boundary ด้วยขอบ
            df[columns] = np.where(df[columns] < lower_bound, lower_bound, df[columns])
            df[columns] = np.where(df[columns] > upper_bound, upper_bound, df[columns])
    return df


def cap_outlier_alternative(df, numerical_feature):
    for column in numerical_feature:
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # ลอง cap outlier ดูก่อน
            capped_values = np.where(df[column] < lower_bound, lower_bound, df[column])
            capped_values = np.where(capped_values > upper_bound, upper_bound, capped_values)

            # เช็คว่าเหลือค่าที่ไม่ซ้ำกันกี่ค่า
            if np.unique(capped_values).size > 1:
                df[column] = capped_values  # ถ้ายังมีมากกว่า 1 ค่า ให้ใช้ค่าที่ cap แล้ว
                df[column] = df[column].apply(np.floor)

    return df

def clean_by_cap_and_fill(df,numerical_feature,categorical_feature,province_json_path):

    cleaned_2_df = df.copy()

    # ---- เติม Numerical Variable ----
    cleaned_2_df = fill_missing_with_median(cleaned_2_df,numerical_feature)

    # ---- Clean Categorical -------
    cleaned_2_df = fill_missing_with_mode(cleaned_2_df, categorical_feature) # เติมด้วย Mode

    # เติมรถคันที่ 1
    priority_list = ['รถยนต์นั่งส่วนบุคคล', 'รถจักรยานยนต์', 'รถปิคอัพโดยสาร', 'รถตู้', 
                    'รถโดยสารมากกว่า4ล้อ', 'รถปิคอัพบรรทุก4ล้อ', 'รถบรรทุก6ล้อ', 
                    'รถบรรทุกไม่เกิน10ล้อ', 'รถบรรทุกมากกว่า10ล้อ', 'รถอีแต๋น', 'อื่นๆ']

    cleaned_2_df['รถคันที่1'] = cleaned_2_df.apply(fill_missing_vehicle, axis=1,priority_list=priority_list)
    cleaned_2_df['รถคันที่1'] = cleaned_2_df['รถคันที่1'].fillna(cleaned_2_df['รถคันที่1'].iloc[0])
    cleaned_2_df = group_categorical_feature(cleaned_2_df) # จัดกลุ่ม


    cleaned_2_df = recalculate_accident_number(cleaned_2_df)

    # # ปรับแก้ Outlier
    # cleaned_2_df_test = cap_outlier(cleaned_2_df,numerical_feature)

    # ปรับแก้ outlier
    cleaned_2_df = cap_outlier_alternative(cleaned_2_df,numerical_feature)

    # Clean ข้อมูลจังหวัด
    cleaned_2_df = fill_province_and_add_region(cleaned_2_df,province_json_path)

    
    return cleaned_2_df


#-------------- Fill Province
def create_province_boundaries(geojson_data):
    """
    Create a dictionary of province boundaries from GeoJSON features
    Returns: dict of {province_name: (boundary_polygon, thai_name)}
    """
    province_boundaries = {}

    for feature in geojson_data['features']:
        try:
            # Get province names
            province_name = feature['properties']['NAME_1']
            # Clean up province name by removing prefixes
            province_name_th = feature['properties']['NL_NAME_1']
            province_name_th = province_name_th.replace('จังหวัด', '').replace('อำเภอเมือง', '').replace('พระนครศรี', '').strip()

            # Create MultiPolygon from coordinates
            coordinates = feature['geometry']['coordinates']
            boundary = MultiPolygon([Polygon(coords[0]) for coords in coordinates])

            # Store both the boundary and Thai name
            province_boundaries[province_name] = (boundary, province_name_th)

        except Exception as e:
            print(f"Error processing province {province_name}: {e}")
            continue

    return province_boundaries

def find_province_for_point(lat, lon, province_boundaries):
    """
    Find which province contains the given point
    Returns: Thai name of the province or empty string if not found
    """
    point = Point(lon, lat)

    for boundary, thai_name in province_boundaries.values():
        if boundary.contains(point):
            return thai_name

    return ""

def fill_province(df, boundary_json):
    """
    Process accident data and determine province for each location
    """
    # Read the GeoJSON data with all province boundaries
    with open(boundary_json, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Create province boundaries dictionary

    province_boundaries = create_province_boundaries(geojson_data)

    # Read the accident data
    # print(f"Processing {total_rows} accident records...")

    # Initialize the province column
    df['province'] = np.nan

    # Process each accident location
    for idx, row in df.iterrows():
        try:
            lat = float(row['LATITUDE'])
            lon = float(row['LONGITUDE'])

            province = find_province_for_point(lat, lon, province_boundaries)
            df.at[idx, 'province'] = province

        except (ValueError, TypeError) as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # no_province = df['province'].isna().sum() + df['province'].eq('').sum()
    df['province'] = df['province'].replace('',np.nan)
    df['จังหวัด'] = df['จังหวัด'].fillna(df['province'])
    df['จังหวัด'] = df['จังหวัด'].fillna(df['จังหวัด'].mode().iloc[0])
    df = df.drop(['province'],axis=1)
    return df


def add_region(df):
    df['ภาค'] =  df['จังหวัด'].apply(get_region)
    return df



# -------------- Prepare Model Data ------------------------
def summarize_columns(df, target_col):
    """
    สรุปประเภทของคอลัมน์ใน DataFrame
    - แยกเป็น numeric, categorical และ target

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการวิเคราะห์
    target_col (str): ชื่อคอลัมน์ที่เป็น Target

    Returns:
    dict: {'numeric': [num_cols], 'categorical': [cat_cols], 'target': target_col}
    """
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in num_cols:
        num_cols.remove(target_col)
    elif target_col in cat_cols:
        cat_cols.remove(target_col)

    return {"numeric": num_cols, "categorical": cat_cols, "target": target_col}


def encode_categorical_features(df, target_col):
    """
    ทำ One-Hot Encoding กับ categorical variables และคืนค่าเป็น DataFrame

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการ encoding
    target_col (str): ชื่อคอลัมน์ที่เป็น Target

    Returns:
    pd.DataFrame: DataFrame ที่ถูก encode แล้ว
    """
    summary = summarize_columns(df, target_col)
    cat_cols = summary['categorical']
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  
    encoded_cat = encoder.fit_transform(df[cat_cols])

    # สร้าง DataFrame ของ One-Hot Encoded Features
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

    # รวมกับ numeric columns และ target
    final_df = pd.concat([df[summary['numeric']], encoded_df, df[target_col]], axis=1)

    return final_df