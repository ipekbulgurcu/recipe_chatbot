import os
import streamlit as st
import textwrap

# V0.1 Sürümlerine Uygun Importlar
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- 1. Yardımcı Fonksiyon: RAG Çekirdeği (Zincirsiz - Önceki Blok) ---
@st.cache_resource
def setup_rag_core():
   
    if "GEMINI_API_KEY" not in os.environ:
        st.error("❌ HATA: GEMINI_API_KEY ayarlanmadı. Lütfen Secrets kısmını kontrol edin.")
        return None, None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
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
        
        return llm, retriever
    except Exception as e:
        st.error(f"❌ HATA: Veri altyapısı kurulurken sorun oluştu: {e}")
        return None, None

# --- 2. Cevap Üretim Fonksiyonu (Generation) ---
def generate_response(llm, retriever, prompt):
    retrieved_docs = retriever.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_prompt_template = textwrap.dedent("""
        Sen harika bir 3000 Yemek Tarifleri Asistanısın. Yalnızca aşağıdaki 'context' 
        içindeki bilgilere dayanarak kullanıcının tarif ve malzeme sorularını Türkçe yanıtla. 
        Cevabında tarifi net ve adım adım açıkla. 
        Eğer tarif mevcut değilse, 'Elimde bu tarife ait bilgi bulunmuyor.' şeklinde kibarca cevap ver.
        
        Context: 
        {context}
    """)
    
    final_prompt = system_prompt_template.format(context=context) + f"\n\nKullanıcı Sorgusu: {prompt}"

    try:
        response = llm.invoke(final_prompt)
        return response.content
    except Exception as e:
        return f"Üzgünüm, Gemini API çağrısı sırasında bir hata oluştu: {e}"

# --- 3. Streamlit Uygulaması ---
st.set_page_config(page_title="3000 Yemek Tarifi Chatbotu (Final Projesi)", layout="wide")
st.title("👨‍🍳 3000 Yemek Tarifi Chatbotu")
st.caption("RAG mimarisi ile 3000 tarife anında ulaşın. (Gemini API + LangChain)")

llm, retriever = setup_rag_core()

if llm and retriever:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hangi yemeğin tarifini istersiniz?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Tarif aranıyor..."):
                full_response = generate_response(llm, retriever, prompt)
                st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

