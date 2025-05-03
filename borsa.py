import streamlit as st

# İlk olarak sayfa yapılandırması yapılmalı
st.set_page_config(
    page_title="Borsa İstanbul Hisse Analiz Paneli",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ana uygulamayı çağır
from ui.main_ui import main

if __name__ == "__main__":
    try:
        # set_page_config diğer dosyalarda devre dışı bırakılmalı
        main()
    except Exception as e:
        st.error(f"Uygulama başlatılırken bir hata oluştu: {str(e)}")
        st.code(f"{type(e).__name__}: {str(e)}") 