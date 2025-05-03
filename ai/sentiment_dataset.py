"""
Türkçe ekonomi, finans ve borsa haberleri için duyarlılık analizi veri seti.
"""

def get_turkish_financial_dataset():
    """
    Türkçe ekonomi, finans ve borsa haberleri için genişletilmiş veri seti döndürür.
    
    Returns:
        tuple: (metinler, etiketler) şeklinde veri seti.
    """
    # Pozitif haberler/metinler
    positive_texts = [
        # Borsa/Hisse
        "Borsa günü yükselişle tamamladı",
        "Borsa İstanbul tarihi rekor kırdı",
        "Borsa haftaya yükselişle başladı",
        "BIST 100 endeksi yüzde 1.5 yükseldi",
        "Hisseler güçlü bir yükseliş gösterdi",
        "Hisse fiyatları tavan yaptı",
        "Yatırımcılar borsada büyük kazanç sağladı",
        "Endeks güne yükselişle başladı",
        "Piyasalar olumlu seyir izledi",
        "Borsada yükseliş trendi devam ediyor",
        "Banka hisseleri yükselişe geçti",
        "Yabancı yatırımcılar Türk hisselerine yöneldi",
        "Şirket hisseleri değer kazandı",
        "BIST rekor tazeledi",
        "Yatırımcılar hisse senedi alımlarını artırdı",
        "Borsa İstanbul tarihi zirvesini gördü",
        "Küçük yatırımcı kazanç elde etti",
        "Yatırımcılar portföylerini büyüttü",
        
        # Ekonomi/Büyüme
        "Türkiye ekonomisi beklentilerden hızlı büyüdü",
        "Ekonomik büyüme rakamları açıklandı, sonuçlar olumlu",
        "GSYH büyüme oranı yüzde 5'i aştı",
        "Ekonomide toparlanma sinyalleri güçleniyor",
        "Sanayi üretiminde güçlü artış",
        "Ekonomik göstergeler iyileşme işaretleri veriyor",
        "İmalat sanayi güçlü performans gösteriyor",
        "İhracat rakamları rekor kırdı",
        "Dış ticaret fazlası vermeye başladık",
        "Türkiye'nin ihracatı artış eğiliminde",
        "İstihdam rakamları iyileşti",
        "İşsizlik oranı geriledi",
        "Ekonomi pozitif büyüme kaydetti",
        "Türkiye'nin kredi notu yükseldi",
        "Enflasyon beklentilerin altında kaldı",
        "Ekonomik reformlar meyvelerini vermeye başladı",
        "Bütçe açığı azaldı",
        "Cari açık düştü",
        
        # Para Piyasaları
        "Türk Lirası değer kazandı",
        "TL diğer para birimleri karşısında güçlendi",
        "Döviz kurlarında gerileme devam ediyor",
        "Enflasyonda düşüş eğilimi başladı",
        "Merkez Bankası faiz indirimine gitti",
        "Kredi faizleri düşüşe geçti",
        "Piyasalarda güven artıyor",
        "Kredi notu artırıldı",
        "Yatırımcı güveni yükseldi",
        "Ekonomik risk primi geriledi",
        "Bankalar ucuz kredi vermeye başladı",
        "Mevduat faizleri yükseldi",
        "Dolar kurunun düşüşü devam ediyor",
        "Euro karşısında TL değer kazandı",
        "Para piyasalarında olumlu hava hakim",
        
        # Şirket Haberleri
        "Şirket karları önceki döneme göre arttı",
        "Şirket büyüme hedeflerini yukarı revize etti",
        "Firmanın satışları beklentileri aştı",
        "İş dünyasında güven endeksi yükseldi",
        "Şirket yeni yatırım kararı aldı",
        "Şirket ihracat anlaşması imzaladı",
        "Şirket hissedarlarına temettü dağıtacak",
        "Şirket yeni istihdam yaratacak",
        "Sirket teknoloji yatırımlarını artırıyor",
        "Satış rakamları geçen yıla göre arttı",
        "Enerji şirketi büyüme hedeflerine ulaştı",
        "Şirket yeni pazarlara açılıyor",
        "Yeni fabrika kuruldu",
        "Otomotiv şirketi satışlarını artırdı",
        "Holding yeni şirket satın aldı",
        "Şirket hisse başına karını artırdı",
        "İhracat rekoru kırıldı",
    ]
    
    # Negatif haberler/metinler
    negative_texts = [
        # Borsa/Hisse
        "Borsa günü sert düşüşle kapattı",
        "Borsa İstanbul sert değer kaybetti",
        "BIST 100 endeksi yüzde 3 geriledi",
        "Hisseler değer kaybetmeye devam ediyor",
        "Borsada satış baskısı sürüyor",
        "Endeks güne düşüşle başladı",
        "Piyasalarda sert satış dalgası yaşanıyor",
        "Borsada düşüş trendi devam ediyor",
        "Yatırımcılar borsada zarar etti",
        "Banka hisseleri sert düştü",
        "Yabancılar Türk hisselerinden çıkıyor",
        "Şirket hisseleri değer kaybetti",
        "BIST sert düşüş yaşadı",
        "Borsada panik satışları yaşandı",
        "Hisse fiyatları dip seviyeleri gördü",
        "Yatırımcılar büyük kayıplar yaşadı",
        "Borsa dibe vurdu",
        "Hisse senetleri eriyor",
        
        # Ekonomi/Daralma
        "Türkiye ekonomisi daraldı",
        "Ekonomik büyüme rakamları beklentilerin altında kaldı",
        "GSYH küçülme gösterdi",
        "Ekonomide yavaşlama sinyalleri artıyor",
        "Sanayi üretiminde düşüş yaşandı",
        "Ekonomik göstergeler kötüleşiyor",
        "İmalat sanayi zayıf performans gösteriyor",
        "İhracat rakamları geriledi",
        "Dış ticaret açığı arttı",
        "Türkiye'nin ihracatı düşüş gösterdi",
        "İstihdam rakamları kötüleşti",
        "İşsizlik oranı yükseldi",
        "Ekonomi negatif büyüme kaydetti",
        "Türkiye'nin kredi notu düşürüldü",
        "Enflasyon beklentilerin üzerinde çıktı",
        "Ekonomik reformlar sonuç vermedi",
        "Bütçe açığı büyüdü",
        "Cari açık arttı",
        
        # Para Piyasaları
        "Türk Lirası değer kaybetti",
        "TL diğer para birimleri karşısında zayıfladı",
        "Döviz kurlarında yükseliş devam ediyor",
        "Enflasyon oranı beklentilerin üzerinde çıktı",
        "Merkez Bankası faiz artışına gitti",
        "Kredi faizleri yükselişe geçti",
        "Piyasalarda güven azalıyor",
        "Kredi notu düşürüldü",
        "Yatırımcı güveni azaldı",
        "Ekonomik risk primi yükseldi",
        "Bankalar kredi vermeyi durdurdu",
        "Mevduat faizleri düştü",
        "Dolar kuru rekor kırdı",
        "Euro karşısında TL değer kaybetti",
        "Para piyasalarında panik havası",
        
        # Şirket Haberleri
        "Şirket karları önceki döneme göre azaldı",
        "Şirket büyüme hedeflerini aşağı revize etti",
        "Firmanın satışları beklentilerin altında kaldı",
        "İş dünyasında güven endeksi düştü",
        "Şirket yatırımlarını durdurdu",
        "Şirket ihracat hedeflerine ulaşamadı",
        "Şirket temettü dağıtmayacak",
        "Şirket personel çıkarmayı planlıyor",
        "Şirket yatırımlarını askıya aldı",
        "Satış rakamları geçen yıla göre düştü",
        "Enerji şirketi zarar açıkladı",
        "Şirket pazardan çekiliyor",
        "Fabrika kapatıldı",
        "Otomotiv şirketi satışlarında düşüş yaşadı",
        "Holding şirket satışını duyurdu",
        "Şirket hisse başına karı düştü",
        "İhracat hedefleri tutturalamadı",
    ]
    
    # Nötr/Belirsiz haberler (negatif olarak etiketleniyor)
    neutral_texts = [
        "Merkez Bankası faiz kararını açıkladı",
        "Ekonomik göstergeler yarın açıklanacak",
        "Şirket gelecek dönem tahminlerini paylaştı",
        "Piyasalar haftaya yatay seyirle başladı",
        "Enflasyon rakamları açıklandı",
        "Borsa İstanbul günü yatay tamamladı",
        "BIST 100 endeksi karışık seyrediyor",
        "Döviz kurlarında dalgalı seyir",
        "Ekonomik paket açıklandı",
        "Yeni ekonomi programı tanıtıldı",
        "Uluslararası kredi kuruluşu rapor yayınladı",
        "Yatırımcılar kararlarını gözden geçiriyor",
        "Hisse senedi analistleri raporlarını paylaştı",
        "Ekonomide belirsizlik sürüyor",
        "Piyasaların yönü belirsiz",
        "Piyasa analistleri görüş bildirdi",
        "Ekonomi yönetimi değerlendirme toplantısı yaptı",
        "TCMB toplantısı gerçekleştirildi",
        "Şirket finansal sonuçlarını açıkladı",
        "Ekonomi zirvesi düzenlendi",
    ]
    
    # Ek pozitif haberler/metinler
    additional_positive = [
        # Borsa/Hisse
        "Borsada alıcılı seyir devam ediyor",
        "Hisse senetleri primli seyrediyor",
        "Borsada boğalar hakimiyeti ele geçirdi",
        "Teknik göstergeler alım sinyali veriyor",
        "Yatırımcıların risk iştahı arttı",
        "Borsa günü artıda kapattı",
        "Haftanın kazandıranları belli oldu",
        "Fon girişleri hızlandı",
        "Piyasada iyimser hava hakim",
        "Alım fırsatları oluştu",
        
        # Ekonomi
        "Ekonomi yönetimi güven verdi",
        "Büyüme tahminleri yukarı yönlü revize edildi",
        "Ekonomik kalkınma hızlandı",
        "Ekonomiye güven artıyor",
        "Teşvik paketi açıklandı",
        "Vergi indirimleri açıklandı",
        "Ekonomide yeşil büyüme dönemi başlıyor",
        "Üretim kapasitesi artıyor",
        "Ekonomide canlanma belirtileri",
        
        # Finans
        "Bankacılık sektörü güçlü performans gösterdi",
        "Finans sektörü büyümeye katkı sağladı",
        "Kredi büyümesi hızlandı",
        "Tasarruf oranları yükseldi",
        "Finansal istikrar güçleniyor",
        "Sigorta şirketleri büyüme kaydetti",
    ]
    
    # Ek negatif haberler/metinler
    additional_negative = [
        # Borsa/Hisse
        "Borsada satış dalgası sürüyor",
        "Hisse senetleri değer kaybediyor",
        "Borsada ayılar hakimiyeti ele geçirdi",
        "Teknik göstergeler satış sinyali veriyor",
        "Yatırımcıların risk iştahı azaldı",
        "Borsa günü eksiye kapattı",
        "Haftanın kaybettirenleri belli oldu",
        "Fon çıkışları hızlandı",
        "Piyasada karamsar hava hakim",
        "Satış baskısı devam ediyor",
        
        # Ekonomi
        "Ekonomi yönetimi güven kaybetti",
        "Büyüme tahminleri aşağı yönlü revize edildi",
        "Ekonomik kalkınma yavaşladı",
        "Ekonomiye güven azalıyor",
        "Destekler yetersiz kaldı",
        "Vergi artışları açıklandı",
        "Ekonomide küçülme dönemi başlıyor",
        "Üretim kapasitesi düşüyor",
        "Ekonomide durgunluk belirtileri",
        
        # Finans
        "Bankacılık sektörü zayıf performans gösterdi",
        "Finans sektörü daralmaya katkı sağladı",
        "Kredi büyümesi yavaşladı",
        "Tasarruf oranları düştü",
        "Finansal kırılganlık artıyor",
        "Sigorta şirketleri küçülme kaydetti",
    ]
    
    # Pozitif ve negatif ek metinleri ana listelere ekle
    positive_texts.extend(additional_positive)
    negative_texts.extend(additional_negative)
    
    # Ek pozitif metin örnekleri - Veri setini dengelemek için
    more_positive = [
        # Daha spesifik ekonomik haberler
        "Sanayi üretimi beklentilerin üstünde gerçekleşti",
        "Kapasite kullanım oranı arttı",
        "İmalat PMI endeksi yükseldi",
        "Büyüme rakamları pozitif yönde revize edildi",
        "İşsizlik oranı son üç yılın en düşük seviyesinde",
        "İhracat rekor kırdı, dış ticaret açığı azaldı",
        "Ekonomik güven endeksi yükseldi",
        "Konut satışları canlandı",
        "Tüketici güven endeksi yükseldi",
        "Perakende satışlar arttı",
        "Cari işlemler fazla verdi",
        "Bütçe gelirleri arttı",
        "Vergi gelirleri beklentilerin üzerinde gerçekleşti",
        
        # Borsa ve hisse senedi ile ilgili haberler
        "Şirket hisseleri yabancı ilgisi ile yükseliyor",
        "Teknoloji şirketleri borsada değer kazandı",
        "Bankacılık sektörü hisseleri ralli yaptı",
        "Enerji şirketleri borsada öne çıktı",
        "Hissede hedef fiyat yükseltildi",
        "Şirket alım sinyali verdi",
        "Güçlü finansallar hisseyi yukarı taşıdı",
        "Portföy yöneticileri alım tavsiyesi verdi",
        "Uzmanlar hisse için 'al' tavsiyeleri veriyor",
        "Şirket tahminlerin üzerinde kar açıkladı",
        "Temettü verimliliği arttı",
        
        # Para piyasaları
        "Merkez Bankası piyasaya likidite sağlayacak",
        "Döviz rezervleri arttı",
        "Faiz indirimi bekleniyor",
        "Swap anlaşması imzalandı",
        "Uluslararası yatırımcı ilgisi arttı",
        "Kredi derecelendirme notu yükseltildi",
        "Tahvil piyasasında yükseliş devam ediyor",
        "Yabancı para girişi hızlandı",
        "Finansal istikrar güçleniyor",
        "Bankalar kredi vermeye başladı",
        "Kredi musluğu açıldı"
    ]
    
    # Pozitif örneklere ekleme yap
    positive_texts.extend(more_positive)
    
    # Daha fazla özel finans ve ekonomi haberleri (pozitif)
    even_more_positive = [
        # Şirket başarı haberleri
        "Şirket pazar payını %15 artırdı",
        "Üretim kapasitesi iki katına çıkarıldı",
        "Ar-Ge yatırımları meyvelerini vermeye başladı",
        "Yeni iş anlaşması imzalandı",
        "Yurtdışı operasyonları büyümeye devam ediyor",
        "Halka arz beklentilerin üzerinde talep gördü",
        "Şirket stratejik ortaklık anlaşması imzaladı",
        "İnovasyon ödülü kazanıldı",
        "Patent başvurusu onaylandı",
        "Marka değeri yükseldi",
        "Şirket değerlemesi arttı",
        "Kârlılık beklentilerin üzerinde gerçekleşti",
        "Operasyonel verimlilik iyileşti",
        "Borç yapılandırması tamamlandı",
        "Finansal sürdürülebilirlik güçlendi",
        
        # Ekonomi politikası haberleri
        "Yeni ekonomik reform paketi açıklandı",
        "Büyüme odaklı politikalar devreye alınıyor",
        "Enflasyonla mücadelede somut adımlar atıldı",
        "Ekonomi yönetimi piyasaları rahatlattı",
        "Mali disiplin güçleniyor",
        "Ekonomik istikrar programı olumlu sonuçlar veriyor",
        "Yapısal reformlar tamamlanıyor",
        "Yatırım ortamı iyileştiriliyor",
        "Finansal düzenlemeler piyasaları olumlu etkiledi",
        "Üretim teşvikleri genişletildi",
        "İhracat destekleri artırıldı",
        "Dijital ekonomiye geçiş hızlanıyor",
        "Yeşil ekonomi yatırımları artıyor",
        "Sürdürülebilir kalkınma hedefleri açıklandı",
        
        # Global ekonomi haberleri - olumlu yansımalar
        "Küresel büyüme Türkiye'yi olumlu etkileyecek",
        "Uluslararası fonlar Türkiye'ye yatırım yapıyor",
        "Doğrudan yabancı yatırımlar arttı",
        "Global ticaret hacmi yükselişe geçti",
        "Dış pazarlarda Türk ürünlerine talep artıyor",
        "Türkiye küresel değer zincirinde öne çıkıyor",
        "Türkiye bölgesel finans merkezi olma yolunda ilerliyor",
        "Uluslararası kredi kuruluşları Türkiye'ye olumlu bakıyor",
        "Türkiye'nin rekabet gücü artıyor",
        "Global yatırımcılar için cazip fırsatlar sunuluyor"
    ]
    
    # Daha fazla negatif örnek ekleyelim
    more_negative = [
        # Detaylı ekonomik sorunlar
        "Enflasyon kalıcı hale geliyor",
        "Stagflasyon riski artıyor",
        "Ekonomide yapısal sorunlar derinleşiyor",
        "Mali göstergeler kötüleşiyor",
        "Ekonomik büyüme sürdürülebilir değil",
        "İşsizlik yapısal hale geldi",
        "Kayıt dışı ekonomi büyüyor",
        "Gelir dağılımı bozuluyor",
        "Ekonomide kırılganlıklar artıyor",
        "Makroekonomik dengesizlikler derinleşiyor",
        "Reel sektör borçları artıyor",
        "Sanayi üretimi çöküşte",
        "Hane halkı borçlanması rekor seviyede",
        "Ekonomide güven endeksleri dip seviyede",
        
        # Şirket sorunları
        "Şirket yeniden yapılandırmaya gidiyor",
        "İflas erteleme başvurusu yapıldı",
        "Şirket küçülme kararı aldı",
        "Üretim hatları kapatılıyor",
        "Şirket pazar payını kaybediyor",
        "Rekabet gücü azalıyor",
        "Teknolojik dönüşüm başarısız oldu",
        "Operasyonel maliyetler kontrolden çıktı",
        "Nakit akışı sorunları baş gösterdi",
        "Satış hedefleri tutturulamadı",
        "Kârlılık oranları düşüyor",
        "Şirket birleşmesi başarısız oldu",
        "Yeniden yapılandırma süreci tıkandı",
        
        # Finans piyasası sorunları
        "Piyasa likiditesi kuruyor",
        "Yabancı sermaye çıkışı hızlandı",
        "Borsa manipülasyon endişeleri artıyor",
        "Marj çağrıları artıyor",
        "Kredi notları düşmeye devam ediyor",
        "Bankacılık sektöründe tahsili gecikmiş alacaklar artıyor",
        "Finans şirketleri kâr uyarısı yaptı",
        "Borsa endeksi teknik desteğin altına indi",
        "Sistemik risk endişeleri artıyor"
    ]
    
    # Veri setine yeni örnekleri ekle
    positive_texts.extend(even_more_positive)
    negative_texts.extend(more_negative)
    
    # Daha fazla finans sektörü ve borsa haberleri (pozitif)
    expert_positive = [
        # Finansal analiz raporları - olumlu
        "Teknik analizler alım fırsatına işaret ediyor",
        "Analistler hisse için 'AL' tavsiyesi verdi",
        "Teknik göstergeler güçlü al sinyali veriyor",
        "Destek seviyelerinden güçlü alımlar geldi",
        "Direnç noktasını aşan hisseler yükselişe geçti",
        "Bollinger bantları alım fırsatı gösteriyor",
        "RSI göstergesi aşırı satım bölgesinden dönüş yaptı",
        "MACD indikatörü alım sinyali verdi",
        "Fibonacci düzeltme seviyeleri destek oluşturdu",
        "Hareketli ortalamalar yukarı yönlü kesişim gösterdi",
        "Momentum göstergeleri pozitif bölgede",
        "Hacim bazlı göstergeler alım baskısını doğruluyor",
        "Düşüş kanalı yukarı yönlü kırıldı",
        "Çift dip formasyonu oluştu, yükseliş bekleniyor",
        "Ters omuz baş omuz formasyonu tamamlandı",
        
        # Şirket finansal sonuçları - olumlu
        "FAVÖK marjı geçen yıla göre artış gösterdi",
        "Şirketin net borç pozisyonu iyileşti",
        "Şirketin brüt kar marjı yükseldi",
        "Faaliyet karı beklentilerin üzerinde gerçekleşti",
        "Şirket piyasa değeri artmaya devam ediyor",
        "Satış gelirlerinde organik büyüme devam ediyor",
        "Şirket net nakit pozisyonuna geçti",
        "İşletme sermayesi ihtiyacı azaldı",
        "Sermaye yeterlilik rasyosu güçlendi",
        "Nakit akışı geçen yıla göre arttı",
        "Gider/gelir oranı iyileşti",
        
        # Ekonomi politikaları - olumlu
        "Merkez Bankası'nın hamleleri piyasalarda güven oluşturdu",
        "Maliye politikası ekonomiyi destekleyici yönde ilerliyor",
        "Para politikası kararları olumlu karşılandı",
        "Ekonomi yönetimi piyasa dostu adımlar atıyor",
        "Makro ihtiyati tedbirler ekonomiyi güçlendiriyor",
        "Yapısal reformlar hız kazandı",
        "Sürdürülebilir büyüme modeli oluşturuluyor",
        "Dış ticaret dengesi iyileşiyor",
        "Cari denge iyileşme gösteriyor",
        "Bütçe disiplini korunuyor",
        "Kamu maliyesi dengeleniyor",
        
        # Sektörel gelişmeler - olumlu
        "Otomotiv sektörü ihracatta rekor kırdı",
        "Enerji sektöründe yeni yatırımlar açıklandı",
        "Teknoloji şirketleri büyümeye devam ediyor",
        "Perakende sektöründe toparlanma başladı",
        "İnşaat sektöründe canlanma sinyalleri",
        "Turizm gelirleri arttı, sektör büyüyor",
        "Tekstil ihracatı yükselişe geçti",
        "Beyaz eşya satışları artıyor",
        "Savunma sanayi ihracatı rekor seviyede"
    ]
    
    # Daha fazla finans sektörü ve borsa haberleri (negatif)
    expert_negative = [
        # Finansal analiz raporları - olumsuz
        "Teknik analizler satış baskısına işaret ediyor",
        "Analistler hisse için 'SAT' tavsiyesi verdi",
        "Teknik göstergeler güçlü sat sinyali veriyor",
        "Destek seviyeleri kırıldı, satış baskısı arttı",
        "Direnç noktasından dönen hisseler düşüşe geçti",
        "Bollinger bantları satış fırsatı gösteriyor",
        "RSI göstergesi aşırı alım bölgesinden dönüş yaptı",
        "MACD indikatörü satım sinyali verdi",
        "Fibonacci düzeltme seviyeleri direnç oluşturdu",
        "Hareketli ortalamalar aşağı yönlü kesişim gösterdi",
        "Momentum göstergeleri negatif bölgede",
        "Hacim bazlı göstergeler satış baskısını doğruluyor",
        "Yükseliş kanalı aşağı yönlü kırıldı",
        "Çift tepe formasyonu oluştu, düşüş bekleniyor",
        "Omuz baş omuz formasyonu tamamlandı",
        
        # Şirket finansal sonuçları - olumsuz
        "FAVÖK marjı geçen yıla göre daralma gösterdi",
        "Şirketin net borç pozisyonu kötüleşti",
        "Şirketin brüt kar marjı geriledi",
        "Faaliyet zararı beklentilerin altında gerçekleşti",
        "Şirket piyasa değeri düşmeye devam ediyor",
        "Satış gelirlerinde daralma yaşanıyor",
        "Şirket net borç pozisyonuna geçti",
        "İşletme sermayesi ihtiyacı arttı",
        "Sermaye yeterlilik rasyosu zayıfladı",
        "Nakit akışı geçen yıla göre azaldı",
        "Gider/gelir oranı bozuldu",
        
        # Ekonomi politikaları - olumsuz
        "Merkez Bankası'nın hamleleri piyasalarda güvensizlik yarattı",
        "Maliye politikası ekonomiyi baskılayıcı yönde ilerliyor",
        "Para politikası kararları olumsuz karşılandı",
        "Ekonomi yönetimi piyasa karşıtı adımlar atıyor",
        "Makro ihtiyati tedbirler yetersiz kalıyor",
        "Yapısal reformlar erteleniyor",
        "Sürdürülemez büyüme modeli endişe yaratıyor",
        "Dış ticaret açığı büyüyor",
        "Cari açık genişliyor",
        "Bütçe disiplininden sapılıyor",
        "Kamu maliyesi bozuluyor",
        
        # Sektörel gelişmeler - olumsuz
        "Otomotiv sektöründe satışlar sert düştü",
        "Enerji sektöründe yeni yatırımlar ertelendi",
        "Teknoloji şirketlerinin kar marjları düşüyor",
        "Perakende sektöründe darboğaz derinleşiyor",
        "İnşaat sektöründe durgunluk sürüyor",
        "Turizm gelirleri düştü, sektör küçülüyor",
        "Tekstil ihracatında daralma yaşanıyor",
        "Beyaz eşya satışları geriliyor",
        "Savunma sanayi projeleri askıya alındı"
    ]
    
    # Önceki olumlu veri setine verilen ağırlık
    for _ in range(2):
        positive_texts.extend(expert_positive)

    # Önceki olumsuz veri setine verilen ağırlık    
    for _ in range(2):
        negative_texts.extend(expert_negative)
    
    # Tüm metinleri birleştir
    all_texts = positive_texts + negative_texts + neutral_texts
    
    # Etiketleri oluştur (1: pozitif, 0: negatif/nötr)
    all_labels = [1] * len(positive_texts) + [0] * (len(negative_texts) + len(neutral_texts))
    
    # Metin uzunlukları kontrol ediliyor
    text_lengths = [len(text) for text in all_texts]
    avg_length = sum(text_lengths) / len(text_lengths)
    
    print(f"Veri seti oluşturuldu: {len(positive_texts)} pozitif, {len(negative_texts) + len(neutral_texts)} negatif örnek")
    print(f"Ortalama metin uzunluğu: {avg_length:.1f} karakter")
    
    return all_texts, all_labels 