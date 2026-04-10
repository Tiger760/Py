import streamlit as st
import pandas as pd
from pandas_datareader import wb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ================================
# Tùy chỉnh Layout Dashboard
# ================================
st.set_page_config(
    page_title="Phân tích Kinh tế Vĩ mô Việt Nam",
    page_icon="📈",
    layout="wide"
)

# Thêm Style cho Dashboard thêm đẹp mắt
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #008000;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 DASHBOARD PHÂN TÍCH VÀ DỰ BÁO KINH TẾ VĨ MÔ VIỆT NAM</p>', unsafe_allow_html=True)
st.markdown('---')

# ================================
# Hàm Load Data (caching)
# ================================
@st.cache_data
def load_data():
    # Load dữ liệu từ WB
    df = wb.download(indicator=['NY.GDP.MKTP.KD.ZG', 'FM.LBL.BMNY.GD.ZS','FP.CPI.TOTL.ZG','FR.INR.RINR','NE.TRD.GNFS.ZS'], 
                     country=['VNM'], start=2010, end=2024)
    # Rút gọn tên cột
    df.columns = ['GDP', 'M2', 'INF','R','TRADE']
    
    # Rút gọn MultiIndex lấy năm
    df.index = df.index.get_level_values('year')
    # Xử lý NaN
    df = df.fillna(df.mean())
    df.reset_index(inplace=True)
    df['year'] = df['year'].astype(int)
    # Sắp xếp tăng dần theo năm
    df = df.sort_values(by='year').reset_index(drop=True)
    return df

df = load_data()

# ================================
# Phần 1: TỔNG QUAN DỮ LIỆU
# ================================
st.markdown('<p class="sub-header">1. Tổng Quan Dữ Liệu World Bank (2010-2024)</p>', unsafe_allow_html=True)
with st.expander("Xem chi tiết bảng số liệu", expanded=False):
    st.dataframe(df.style.format(precision=2), use_container_width=True)

# ================================
# Phần 2: BIỂU ĐỒ TRỰC QUAN HÓA (EDA)
# ================================
st.markdown('<p class="sub-header">2. Phân Tích Chỉ Số (Exploratory Data Analysis)</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# ----- Biểu đồ GDP -----
with col1:
    fig_gdp = px.bar(df, x='year', y='GDP', text_auto='.2f',
                     title='💰 Tốc độ Tăng Trưởng GDP Việt Nam (%)',
                     labels={'year':'Năm', 'GDP':'Tăng trưởng GDP (%)'},
                     color_discrete_sequence=['#ff7f50'])
    fig_gdp.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_gdp, use_container_width=True)

# ----- Biểu đồ M2 và GDP hoặc Lạm Phát -----
with col2:
    fig_inf = go.Figure()
    fig_inf.add_trace(go.Scatter(x=df['year'], y=df['GDP'], mode='lines+markers', name='GDP (%)',
                                 line=dict(color='#ff7f50', width=3), marker=dict(size=8)))
    fig_inf.add_trace(go.Scatter(x=df['year'], y=df['INF'], mode='lines+markers', name='Lạm phát (INF %)',
                                 line=dict(color='#800000', width=3), marker=dict(size=8)))
    fig_inf.update_layout(title='⚖️ Tương quan Biến động giữa GDP và Lạm Phát',
                          xaxis_title='Năm', yaxis_title='Phần trăm (%)', hovermode="x unified")
    st.plotly_chart(fig_inf, use_container_width=True)


col3, col4 = st.columns([1, 1])

# ----- Biểu đồ Heatmap Tương quan -----
with col3:
    st.markdown("<h4 style='text-align: center;'>🔥 Ma Trận Tương Quan Các Chỉ Số</h4>", unsafe_allow_html=True)
    # Loại cột year ra để tính correlation
    df_corr = df.drop(columns=['year']).corr()
    
    fig_heat = px.imshow(df_corr, text_auto=".2f", aspect="auto", 
                         color_continuous_scale='RdYlGn', origin='lower')
    st.plotly_chart(fig_heat, use_container_width=True)

with col4:
    st.markdown("<h4 style='text-align: center;'>📊 Xu Hướng Độ Mở Kinh Tế (Trade/GDP)</h4>", unsafe_allow_html=True)
    fig_trade = px.area(df, x='year', y='TRADE',
                     labels={'year':'Năm', 'TRADE':'Giá trị Ngoại thương so với GDP (%)'},
                     color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_trade, use_container_width=True)


# ================================
# Phần 3: MÔ HÌNH DỰ ĐOÁN LINEAR REGRESSION
# ================================
st.markdown('<p class="sub-header">3. Dự Phóng Tăng Trưởng Kinh Tế (Machine Learning Prediction)</p>', unsafe_allow_html=True)
st.write("Sử dụng thuật toán **Linear Regression** (Hồi quy tuyến tính Đa biến) tìm ra mức tăng trưởng tối ưu từ các biến đầu vào.")

# Train mô hình học máy
X = df[['M2', 'INF', 'R', 'TRADE']]
y = df['GDP']
model = LinearRegression()
model.fit(X, y)

# Thêm Logo Cục Phòng Chống Rửa Tiền / Ngân hàng Nhà Nước Việt Nam
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("<h4 style='text-align: center; color: #008000;'>CỤC PHÒNG, CHỐNG RỬA TIỀN</h4>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Khởi tạo FORM bên Sidebar để nhập liệu tương tác
st.sidebar.markdown("## ⚙️ BẢNG ĐIỀU KHIỂN DỰ BÁO")
st.sidebar.markdown("Điều chỉnh các chỉ số kinh tế vĩ mô để dự báo tác động.")

input_M2 = st.sidebar.slider("Tiền cơ sở M2 (% of GDP)", 
                             min_value=50.0, max_value=250.0, value=float(df['M2'].mean()), step=1.0)
input_INF = st.sidebar.number_input("Lạm phát - INF (%)", 
                                    min_value=-5.0, max_value=30.0, value=float(df['INF'].mean()), step=0.1)
input_R = st.sidebar.slider("Lãi suất thực - R (%)", 
                            min_value=-25.0, max_value=15.0, value=float(df['R'].mean()), step=0.5)
input_TRADE = st.sidebar.slider("Thương mại (TRADE as % of GDP)", 
                                min_value=100.0, max_value=250.0, value=float(df['TRADE'].mean()), step=1.0)

# Dự đoán dữ liệu
predict_m2 = round(input_M2, 2)
predict_inf = round(input_INF, 2)
predict_r = round(input_R, 2)
predict_t = round(input_TRADE, 2)

y_predict = model.predict([[predict_m2, predict_inf, predict_r, predict_t]])

# Hiển thị kết quả ra HTML trực quan
st.markdown("### Kết Quả Output")
st.info(f"Dựa trên cấu hình: **M2 = {predict_m2}% | INF = {predict_inf}% | R = {predict_r}% | TRADE = {predict_t}%**")

gdp_rate = round(float(y_predict[0]), 2)
color = "green" if gdp_rate > 5 else "orange" if gdp_rate > 0 else "red"

st.markdown(f"""
<div style='background-color:#f0f2f6; padding: 20px; border-radius: 10px; text-align:center;'>
    <h3>DỰ BÁO TỐC ĐỘ TĂNG TRƯỞNG GDP LÀ:</h3>
    <h1 style='color: {color}; font-size: 50px;'>{gdp_rate} %</h1>
</div>
""", unsafe_allow_html=True)
