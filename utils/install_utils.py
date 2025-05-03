import pip
import subprocess
import importlib
import sys
import logging # Loglama için
import streamlit as st # st.warning/error/info kullanmak için
import types # Monkey patch için

# Logging ayarları
# logger = logging.getLogger(__name__)
# Streamlit logları yeterli olabilir, ekstra logger'a gerek yok gibi

def install_package(package):
    """
    Belirtilen paketi pip ile yükler
    """
    st.info(f"{package} paketi yükleniyor... Bu biraz zaman alabilir.")
    try:
        # subprocess yerine pip.main kullanılabilir
        result = pip.main(['install', package])
        if result == 0:
            st.success(f"{package} paketi başarıyla yüklendi.")
            return True
        else:
            st.error(f"{package} paketi yüklenemedi. Hata kodu: {result}")
            return False
    except Exception as e:
        st.error(f"{package} paketi yüklenirken hata oluştu: {str(e)}")
        return False

def ensure_news_libraries():
    """
    Haber API'leri için gerekli kütüphanelerin yüklü olduğundan emin olur ve monkey patch uygular
    """
    required_packages = ['pygooglenews', 'newspaper3k', 'lxml_html_clean', 'bs4']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'newspaper3k':
                importlib.import_module('newspaper')
            elif package == 'lxml_html_clean':
                importlib.import_module('lxml_html_clean')
            elif package == 'bs4':
                importlib.import_module('bs4')
            else:
                importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"Haber özelliğini kullanmak için gerekli kütüphaneler ({', '.join(missing_packages)}) eksik. Yükleniyor...")
        for package in missing_packages:
            try:
                install_package(package)
            except Exception as e:
                st.error(f"{package} yüklenirken hata oluştu: {str(e)}")
                return False
            
        st.info("Kurulum tamamlandı. Sayfa yenileniyor...")
        
        # Monkey patch'i uygula
        try:
            import lxml_html_clean
            
            # lxml.html için sahte bir modül oluşturun
            if 'lxml.html.clean' in sys.modules:
                del sys.modules['lxml.html.clean']
            
            if 'lxml' not in sys.modules:
                sys.modules['lxml'] = types.ModuleType('lxml')
            
            if 'lxml.html' not in sys.modules:
                sys.modules['lxml.html'] = types.ModuleType('lxml.html')
            
            # lxml.html.clean modülünü lxml_html_clean ile değiştirin
            sys.modules['lxml.html.clean'] = lxml_html_clean
            
            st.success("lxml.html.clean başarıyla lxml_html_clean ile değiştirildi.")
        except Exception as e:
            st.warning(f"Monkey patch uygulanamadı: {str(e)}")
            
        st.rerun()  # Sayfayı yenile
        
    return True

def ensure_ai_libraries():
    """
    Yapay zeka için gerekli kütüphanelerin yüklü olduğundan emin olur
    """
    required_packages = ['google-generativeai']
    missing_packages = []
    
    # Google GenerativeAI kontrolü
    try:
        import google.generativeai
    except ImportError:
        missing_packages = required_packages
    
    if missing_packages:
        st.warning(f"Yapay zeka özelliği için gerekli kütüphaneler ({', '.join(missing_packages)}) eksik. Yükleniyor...")
        for package in missing_packages:
            try:
                install_package(package)
                # Yükleme sonrası import'u kontrol et
                try:
                    import google.generativeai
                    st.success(f"{package} başarıyla yüklendi ve içe aktarıldı.")
                except ImportError:
                    st.error(f"{package} yüklendi ancak içe aktarılamadı! Python sürümü uyumsuz olabilir.")
                    return False
            except Exception as e:
                st.error(f"{package} yüklenirken hata oluştu: {str(e)}")
                return False
            
        st.info("Kurulum tamamlandı. Sayfa yenileniyor...")
        st.rerun()  # Sayfayı yenile
        
    return True 