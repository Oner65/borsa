from data.db_utils import create_database
import sys
import traceback

print("Veritabanı oluşturuluyor...")
try:
    result = create_database()
    print(f"Veritabanı oluşturma işlemi: {'Başarılı' if result else 'Başarısız'}")
    print("NOT: Eğer ml_models tablosunun oluşturulduğundan emin olmak istiyorsanız,")
    print("data/stock_analysis.db dosyasını silip bu scripti tekrar çalıştırabilirsiniz.")
except Exception as e:
    print(f"Veritabanı oluşturulurken hata: {str(e)}")
    traceback.print_exc() 