import numpy as np

import pandas as pd

from datetime import datetime

def preprocess(data):
    
    data.drop_duplicates(inplace=True)

    brands = ['Apple', 'Asus', 'MSI', 'Lenovo', 'Acer', 'Samsung', 'HP', 'Dell']
    data['Brand'] = data['Brand'].apply(lambda x : x if x in brands else 'Other')


    current_year = datetime.now().year 
    data['Laptop_Age'] = current_year - data['Launch_Year'] 
    data.drop(columns=['Launch_Year'], inplace=True)


    data['CPU_Series'] = 'low'
    mid_series_CPU = 'i5|Ultra 5|Core 5| Core 7|Ryzen 5|Ryzen 5 PRO|Ryzen AI 5|Ryzen AI 5 PRO|M5|M4|M3|X Plus'
    data.loc[data['CPU_Model'].str.contains(mid_series_CPU, case=False, na=False), 'CPU_Series'] = 'mid'
    high_series_CPU = 'i7|i9|Ultra 7|Ultra 9|Ryzen 7| Ryzen 7 Pro|Ryzen 9|Ryzen AI 7|Ryzen AI 9|Ryzen AI MAX|Ryzen AI MAX+|M5 Pro|M5 MAX|M4 Pro|M4 MAX|M3 Pro|M3 MAX|X Elite'
    data.loc[data['CPU_Model'].str.contains(high_series_CPU, case=False, na=False), 'CPU_Series'] = 'high'
    
    data['CPU_Segment'] = 'mid'
    high_segment_CPU = 'H|HX|HS|Max|Pro|Elite'
    data.loc[data['CPU_Model'].str.contains(high_segment_CPU, case=False, na=False), 'CPU_Segment'] = 'high'
    data.loc[data['CPU_Model'].str.endswith('U'), 'CPU_Segment'] = 'low'
    data.loc[data['CPU_Model'].str.contains('Pentium|Celeron', case=False, na=False), 'CPU_Segment'] = 'low'

    data['CPU_Generation'] = 'modern'
    latest_gen_CPU = 'Snapdragon|M5|AI|Core Ultra'
    data.loc[data['CPU_Model'].str.contains(latest_gen_CPU, case=False, na=False), 'CPU_Generation'] = 'latest'

    data.drop(columns=['CPU_Model'], inplace=True)


    data['GPU_Tier'] = 'low'
    high_tier_GPU = 'RTX 4070|RTX 4070 Ti|RTX 4080|RTX 4090|RTX 5070|RTX 5070 Ti|RTX 5080|RTX 5090|M5 MAX|M4 MAX|M3 MAX|M5 Pro|M4 Pro|M3 Pro'
    data.loc[data['GPU_Model'].str.contains(high_tier_GPU, case=False, na=False), 'GPU_Tier'] = 'high'
    mid_tier_GPU = 'RTX 5060|RTX 5050|RTX 4060|RTX 4050|RTX 3060|RTX 3050|Intel Arc|660M|680M|760M|780M|840M|860M|880M|890M'
    data.loc[data['GPU_Model'].str.contains(mid_tier_GPU, case=False, na=False), 'GPU_Tier'] = 'mid'

    data.drop(columns=['GPU_Model'], inplace=True)


    data['GPU_VRAM'] = data['GPU_VRAM'].str.replace('GB', '').str.strip()
    data['GPU_VRAM'] = data['GPU_VRAM'].replace('Shared', '0')
    data['GPU_VRAM'] = data['GPU_VRAM'].astype(int)
    

    data[['ScreenW', 'ScreenH']] = data['Resolution'].str.split('x', expand=True).astype(int)
    data['Pixel_Per_Inch'] = (np.sqrt(np.square(data['ScreenW']) + np.square(data['ScreenH'])) / data['Screen_Size']).round(2)
    data.drop(columns=['Screen_Size', 'Resolution', 'ScreenW', 'ScreenH'], inplace=True)

    data.drop(columns=['Storage_Type'], inplace=True)

    return data 
