# recipe_chatbot

# Akbank GenAI Bootcamp: 3000 Yemek Tarifi RAG Chatbotu

Bu proje, RAG (Retrieval Augmented Generation) mimarisi kullanılarak geliştirilmiş, geniş bir yemek tarifi veri setine dayanan yapay zeka destekli bir sohbet robotudur.

## 1. Projenin Amacı
Projenin temel amacı, bir Large Language Model'in (LLM) dışarıdan sağlanan özel bir bilgi setine (yemek tarifleri) erişimini sağlayarak, kullanıcıların spesifik sorularına halüsinasyon riski olmadan doğru ve bağlamsal cevaplar üretmektir.

## 2. Veri Seti Hakkında Bilgi
Projede kullanılan veri seti, 3000'den fazla farklı yemek tarifini içeren **`3000 Yemek Tarifi.txt`** adlı metin dosyasıdır. Bu veri seti, geniş bir mutfak yelpazesinden tarif bilgisi sunarak, chatbot'un bilgi tabanını oluşturmaktadır.

## 3. Kullanılan Yöntemler ve Çözüm Mimarisi (RAG)
Projemiz, aşağıdaki bileşenlerden oluşan bir RAG mimarisi üzerine kuruludur:
1.  **Veri Yükleme ve Parçalama:** `TextLoader` ile metin verisi yüklenir ve `RecursiveCharacterTextSplitter` ile 1500'er karakterlik parçalara (chunks) ayrılır.
2.  **Vektörleştirme (Embedding):** Parçalar, `GoogleGenerativeAIEmbeddings` (`text-embedding-004` modeli) kullanılarak sayısal vektörlere dönüştürülür.
3.  **Vektör Veritabanı:** Vektörler, `ChromaDB` içerisinde saklanır.
4.  **Üretim (Generation):** Kullanıcı sorgusu geldiğinde, ilgili tarif parçaları Vektör DB'den çekilir ve `ChatGoogleGenerativeAI` (`gemini-2.5-flash` modeli) ile birleştirilerek nihai cevap üretilir.

## 4. Elde Edilen Sonuçlar
* Chatbot, **3000** farklı tariften gelen sorulara, kaynak metne dayalı olarak hızlı ve bağlamsal cevaplar verebilmektedir.
* Halüsinasyon riski, RAG mimarisi sayesinde en aza indirilmiştir.

## 5. Çalışma Kılavuzu ve Kurulum (Opsiyonel Bölüm)
Lokalde çalıştırmak için: `python -m venv venv`, `.\venv\Scripts\activate` ve `pip install -r requirements.txt` adımlarını takip edin. Ardından `set GEMINI_API_KEY="YOUR_API_KEY"` komutunu ayarlayarak `streamlit run app.py` ile uygulamayı başlatın.

---
### 🌐 Web Linki (Deploy Linki)

Projenin çalışan versiyonuna aşağıdaki linkten erişebilirsiniz:

**[DEPLOY LİNKİNİZ BURAYA GELECEK]**
