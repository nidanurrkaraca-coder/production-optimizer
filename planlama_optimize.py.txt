import streamlit as st
import pandas as pd
import pulp
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# ------------------- Sayfa Yapılandırması -------------------
st.set_page_config(
    page_title="Üretim Planlama Optimizasyon Sistemi",
    page_icon="🏭",
    layout="wide"
)

# ------------------- Başlık -------------------
st.title("🏭 Üretim Planlama Optimizasyon Sistemi")
st.markdown("**Personel sayısına göre haftalık üretim dağılımını optimize eder (Fazla mesaiyi minimize eder).**")
st.divider()

# ------------------- Optimizasyon Fonksiyonu -------------------
def run_optimization(df_raw):
    try:
        df_raw.columns = ['col0', 'ÜRÜN', 'ATÖLYE', 'SÜRE', 'TOPLAM ADET',
                          '1. HAFTA', '2. HAFTA', '3. HAFTA', '4. HAFTA', '5. HAFTA',
                          'col10', 'col11', 'col12', 'Atölye', 'Personel Sayısı']
        
        products = df_raw[df_raw['ÜRÜN'].str.startswith('A', na=False)].copy()
        products['SÜRE'] = pd.to_numeric(products['SÜRE'], errors='coerce')
        products['TOPLAM ADET'] = pd.to_numeric(products['TOPLAM ADET'], errors='coerce')
        products = products.dropna(subset=['SÜRE', 'TOPLAM ADET'])
        products['TOPLAM ADET'] = products['TOPLAM ADET'].astype(int)
        
        if products.empty:
            st.error("❌ Ürün verileri bulunamadı!")
            return None, None, None, None, None
        
        personnel_raw = df_raw[df_raw['Atölye'].str.startswith('AĞIR KAYNAK', na=False)].copy()
        personnel_raw['Personel Sayısı'] = pd.to_numeric(personnel_raw['Personel Sayısı'], errors='coerce')
        personnel_raw = personnel_raw.dropna(subset=['Personel Sayısı'])
        personnel_dict = dict(zip(personnel_raw['Atölye'], personnel_raw['Personel Sayısı'].astype(int)))
        
        if not personnel_dict:
            st.error("❌ Personel verileri bulunamadı!")
            return None, None, None, None, None
        
        weeks = ['2. HAFTA', '3. HAFTA', '4. HAFTA', '5. HAFTA']
        daily_minutes = 540
        days_per_week = 5
        weekly_capacity = daily_minutes * days_per_week
        
        model = pulp.LpProblem("Uretim_Planlama", pulp.LpMinimize)
        
        x = {}
        for idx, row in products.iterrows():
            product = row['ÜRÜN']
            for w in weeks:
                x[(product, w)] = pulp.LpVariable(f"x_{product}_{w}", lowBound=0, cat='Integer')
        
        Z = pulp.LpVariable("Z", lowBound=0, cat='Continuous')
        overtime = {}
        for atolye in personnel_dict.keys():
            for w in weeks:
                overtime[(atolye, w)] = pulp.LpVariable(f"OT_{atolye}_{w}", lowBound=0, cat='Continuous')
        
        M = 10000
        model += Z + M * pulp.lpSum(overtime.values())
        
        for idx, row in products.iterrows():
            product = row['ÜRÜN']
            total_demand = row['TOPLAM ADET']
            model += pulp.lpSum([x[(product, w)] for w in weeks]) == total_demand
        
        for idx, row in products.iterrows():
            product = row['ÜRÜN']
            for w in weeks:
                model += x[(product, w)] <= Z
        
        for atolye, personel_sayisi in personnel_dict.items():
            atolye_products = products[products['ATÖLYE'] == atolye]
            if atolye_products.empty:
                continue
            for w in weeks:
                workload = pulp.lpSum([
                    atolye_products.loc[idx, 'SÜRE'] * x[(atolye_products.loc[idx, 'ÜRÜN'], w)]
                    for idx in atolye_products.index
                ])
                max_capacity = personel_sayisi * weekly_capacity
                model += workload <= max_capacity + overtime[(atolye, w)]
        
        with st.spinner('🧠 Optimizasyon çalışıyor... Lütfen bekleyin.'):
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
            model.solve(solver)
        
        status = pulp.LpStatus[model.status]
        
        result_df = products[['ÜRÜN', 'ATÖLYE', 'SÜRE', 'TOPLAM ADET']].copy()
        for w in weeks:
            result_df[w] = result_df['ÜRÜN'].apply(lambda p: int(pulp.value(x[(p, w)]) or 0))
        result_df['KONTROL_TOP'] = result_df[weeks].sum(axis=1)
        
        ot_df = pd.DataFrame(index=personnel_dict.keys(), columns=weeks)
        for atolye in personnel_dict.keys():
            for w in weeks:
                val = pulp.value(overtime[(atolye, w)]) or 0
                ot_df.loc[atolye, w] = int(round(val))
        
        original_plan = products[['ÜRÜN', 'ATÖLYE', 'SÜRE', 'TOPLAM ADET'] + weeks].copy()
        
        # Kapasite analizi
        workload_data = []
        for atolye in personnel_dict.keys():
            atolye_products = products[products['ATÖLYE'] == atolye]
            if atolye_products.empty:
                continue
            for w in weeks:
                total_work = sum(
                    atolye_products.loc[idx, 'SÜRE'] * result_df.loc[idx, w]
                    for idx in atolye_products.index
                )
                used_personnel = total_work / weekly_capacity
                mevcut_personel = personnel_dict[atolye]
                diff_personnel = mevcut_personel - used_personnel
                ot_min = ot_df.loc[atolye, w]
                
                workload_data.append({
                    'Atölye': atolye,
                    'Hafta': w,
                    'Toplam İş Yükü (dk)': total_work,
                    'Kullanılan Personel': round(used_personnel, 2),
                    'Mevcut Personel': mevcut_personel,
                    'Personel Farkı': round(diff_personnel, 2),
                    'Fazla Mesai (dk)': ot_min
                })
        
        workload_df = pd.DataFrame(workload_data)
        workload_df['Kapasite Kullanım %'] = (workload_df['Kullanılan Personel'] / workload_df['Mevcut Personel'] * 100).round(1)
        
        return result_df, ot_df, original_plan, personnel_dict, status, workload_df
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        return None, None, None, None, None, None

def create_excel_download(result_df, ot_df, original_plan, personnel_dict, workload_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='OPTIMIZE_PLAN', index=False)
        ot_df.to_excel(writer, sheet_name='FAZLA_MESAI')
        original_plan.to_excel(writer, sheet_name='ORIJINAL_PLAN', index=False)
        pers_df = pd.DataFrame(list(personnel_dict.items()), columns=['Atölye', 'Personel Sayısı'])
        pers_df.to_excel(writer, sheet_name='PERSONEL', index=False)
        workload_df.to_excel(writer, sheet_name='KAPASITE_ANALIZI', index=False)
    return output.getvalue()

# ------------------- ANA ARAYÜZ -------------------
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📂 Dosya Yükle")
    uploaded_file = st.file_uploader(
        "Excel dosyasını seçin (format: .xlsx)",
        type=["xlsx"],
        help="Dosya 'Sayfa1' sayfasını ve doğru sütun başlıklarını içermelidir."
    )
    
    if uploaded_file is not None:
        st.success("✅ Dosya başarıyla yüklendi!")
        if st.button("🚀 Sistemi Başlat", type="primary", use_container_width=True):
            st.session_state['run_optimization'] = True
    else:
        st.info("📤 Lütfen bir dosya yükleyin.")

with col2:
    st.subheader("📊 Optimizasyon Sonuçları")
    
    if uploaded_file is not None and st.session_state.get('run_optimization', False):
        try:
            df_raw = pd.read_excel(uploaded_file, sheet_name="Sayfa1", header=1)
            result, ot, original, personnel, status, workload = run_optimization(df_raw)
            
            if result is not None:
                st.success(f"✅ Optimizasyon tamamlandı! Durum: {status}")
                
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("📦 Toplam Ürün", len(result))
                with col_met2:
                    total_ot = ot.values.sum()
                    st.metric("⏰ Toplam Fazla Mesai (dk)", f"{int(total_ot):,}")
                with col_met3:
                    st.metric("🏭 Atölye Sayısı", len(personnel))
                with col_met4:
                    avg_util = workload['Kapasite Kullanım %'].mean()
                    st.metric("📈 Ort. Kapasite Kullanımı", f"{avg_util:.1f}%")
                
                # Grafik
                st.subheader("📊 Atölye Bazında Haftalık Personel Kullanımı")
                fig = px.bar(workload, x='Atölye', y='Kullanılan Personel', color='Hafta', 
                             barmode='group', text_auto=True,
                             title="Haftalık Kullanılan Personel (Mevcut Personel ile Karşılaştırma)")
                for atolye in personnel.keys():
                    fig.add_hline(y=personnel[atolye], line_dash="dash", 
                                  line_color="red", opacity=0.5,
                                  annotation_text=f"Mevcut: {personnel[atolye]}", 
                                  annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)
                
                # Kapasite Detay Tablosu
                st.subheader("📋 Haftalık Kapasite Detayları")
                styled_workload = workload.style.applymap(
                    lambda val: 'background-color: #d4edda; color: #155724;' if val > 0 else ('background-color: #f8d7da; color: #721c24;' if val < 0 else ''),
                    subset=['Personel Farkı']
                )
                st.dataframe(styled_workload, use_container_width=True)
                
                # Hafta Özetleri
                st.subheader("📌 Haftalık Özet Bilgiler")
                for hafta in ['2. HAFTA', '3. HAFTA', '4. HAFTA', '5. HAFTA']:
                    hafta_data = workload[workload['Hafta'] == hafta]
                    if not hafta_data.empty:
                        total_personel_kullanilan = hafta_data['Kullanılan Personel'].sum()
                        toplam_mevcut = hafta_data['Mevcut Personel'].sum()
                        toplam_fark = total_personel_kullanilan - toplam_mevcut
                        toplam_ot = hafta_data['Fazla Mesai (dk)'].sum()
                        
                        with st.expander(f"📅 {hafta} Detayları"):
                            st.write(f"**Toplam Kullanılan Personel:** {total_personel_kullanilan:.2f}")
                            st.write(f"**Toplam Mevcut Personel:** {toplam_mevcut}")
                            if toplam_fark > 0:
                                st.warning(f"⚠️ Bu hafta {toplam_fark:.2f} kişi **ek personel** ihtiyacı var.")
                            elif toplam_fark < 0:
                                st.success(f"✅ Bu hafta {abs(toplam_fark):.2f} kişi **boş kapasite** var.")
                            else:
                                st.info("⚖️ Personel dengesi tam.")
                            if toplam_ot > 0:
                                st.warning(f"⏰ Toplam fazla mesai: {int(toplam_ot)} dakika.")
                            else:
                                st.success("✅ Fazla mesai yok.")
                            st.dataframe(hafta_data[['Atölye', 'Kullanılan Personel', 'Mevcut Personel', 
                                                      'Personel Farkı', 'Fazla Mesai (dk)', 'Kapasite Kullanım %']],
                                         use_container_width=True)
                
                # Sekmeler
                tab1, tab2, tab3 = st.tabs(["📋 Optimize Plan", "⏱️ Fazla Mesai", "📄 Orijinal Plan"])
                with tab1:
                    st.dataframe(result, use_container_width=True, height=400)
                with tab2:
                    st.dataframe(ot, use_container_width=True)
                with tab3:
                    st.dataframe(original, use_container_width=True, height=400)
                
                # İndirme
                excel_data = create_excel_download(result, ot, original, personnel, workload)
                st.download_button(
                    label="📥 Optimize Edilmiş Excel'i İndir",
                    data=excel_data,
                    file_name="Planlama_Optimize.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                st.session_state['run_optimization'] = False
                
        except Exception as e:
            st.error(f"❌ Dosya okunamadı veya işlenemedi: {str(e)}")
            st.session_state['run_optimization'] = False
    
    elif uploaded_file is None:
        st.warning("⏳ Lütfen sol menüden bir dosya seçin.")

st.divider()
st.caption("📌 Varsayımlar: 5 iş günü, günlük 540 dk (30 dk yemek + 15+15 dk ihtiyaç molası).")