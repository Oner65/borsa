import streamlit as st
from ui.main_ui import main

# Ana uygulamayı başlat
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Uygulama başlatılırken bir hata oluştu: {str(e)}")
        st.code(f"{type(e).__name__}: {str(e)}")