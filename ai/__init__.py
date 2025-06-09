"""
Yapay zeka ve makine öğrenimi modülleri
"""

# Temel modülleri import et
try:
    from . import sentiment_analysis
except ImportError:
    pass

# Streamlit gerektiren modülleri koşullu import et
try:
    import streamlit
    from . import api
except ImportError:
    # Streamlit yüklü değilse api modülünü yükleme
    pass

try:
    from . import predictions
except ImportError:
    pass

__all__ = ['sentiment_analysis', 'predictions', 'api'] 