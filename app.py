import os
import streamlit as st

# LangChain V0.2.x yerine V0.3.x+ uyumluluğu için importları güncelliyoruz
from langchain import globals  # Gerekli olabilir
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document  # Document sınıfı için

# Google Modelleri
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Topluluk Bileşenleri (Bu, hatalı kısmı aşar)
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import Chroma

# --- 1. Yardımcı Fonksiyon: RAG Zincirini Başlatma (Hafızada tutmak için) ---
@st.cache_resource
def setup_rag_chain():
    # API Anahtar Kontrolü (Streamlit Secrets'ı kullanmak en güvenli yoldur)
    if "GEMINI_API_KEY" not in os.environ:
        st.error("❌ HATA: GEMINI_API_KEY ortam değişkeni veya Streamlit Secret ayarlanmadı. Lütfen ayarlayın.")
        return None

    # Modelleri başlat
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    # Veri Yükleme ve Vektör Veritabanı (Çözüm Mimariniz)
    file_path = "3000 Yemek Tarifi.txt"
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=["TARİF:", "YAPILIŞ TARİFİ", "MALZEMELER", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Prompt Şablonu
        system_prompt = (
            "Sen harika bir 3000 Yemek Tarifleri Asistanısın. Yalnızca aşağıdaki 'context' "
            "içindeki bilgilere dayanarak kullanıcının tarif ve malzeme sorularını Türkçe yanıtla. "
            "Cevabında tarifi net ve adım adım açıkla. "
            "Eğer tarif mevcut değilse, 'Elimde bu tarife ait bilgi bulunmuyor.' şeklinde kibarca cevap ver."
            "\n\nContext: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        return rag_chain

    except FileNotFoundError:
        st.error(f"❌ HATA: '{file_path}' dosyası bulunamadı. Lütfen Streamlit uygulamasının yanına koyun.")
        return None
    except Exception as e:
        st.error(f"❌ HATA: RAG zinciri başlatılamadı: {e}")
        return None

# --- 2. Streamlit Uygulaması ---
st.set_page_config(page_title="3000 Yemek Tarifi Chatbotu (GenAI Projesi)", layout="wide")
st.title("👨‍🍳 3000 Yemek Tarifi Chatbotu")
st.caption("RAG mimarisi ile 3000 tarife anında ulaşın. (Gemini API + LangChain)")

rag_chain = setup_rag_chain()

if rag_chain:
    # Sohbet geçmişini başlat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları görüntüle
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al ve RAG'ı çalıştır
    if prompt := st.chat_input("Hangi yemeğin tarifini istersiniz?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Tarif aranıyor..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    full_response = response['answer']
                    st.markdown(full_response)
                except Exception as e:
                    full_response = "Üzgünüm, bir hata oluştu. Lütfen Gemini API anahtarınızı kontrol edin."
                    st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        
