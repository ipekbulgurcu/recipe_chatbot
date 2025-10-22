import os
import gradio as gr
import textwrap

# V0.1 Sürümlerine Uyumlu Importlar
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- 1. RAG Çekirdeğini Kurma ---
def setup_rag_core():
    # API Anahtar Kontrolü (Hugging Face'de Secrets olarak ayarlanmalı)
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmadı. Lütfen Hugging Face Secrets'e ekleyin.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # Veri Yükleme ve Vektör Veritabanı
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
        # Uygulamanın başlatılmasını engeller, hata mesajını gösterir
        raise Exception(f"RAG altyapısı kurulurken kritik hata: {e}")

# Uygulama başlangıcında RAG altyapısını kur
try:
    LLM_MODEL, RETRIEVER = setup_rag_core()
    INITIAL_MESSAGE = "RAG chatbot hazır. Hangi yemeğin tarifini istersiniz?"
except ValueError as e:
    LLM_MODEL, RETRIEVER = None, None
    INITIAL_MESSAGE = f"HATA: API Anahtarı Ayarlanmadı. Lütfen Hugging Face Secrets'ı kontrol edin. ({e})"
except Exception as e:
    LLM_MODEL, RETRIEVER = None, None
    INITIAL_MESSAGE = f"HATA: Uygulama Başlatılamadı. {e}"


# --- 3. Cevap Üretim Fonksiyonu ---
def generate_gradio_response(history, prompt):
    if not LLM_MODEL:
        return [(prompt, INITIAL_MESSAGE)] # API hatasını döndür

    # 1. Retrieval (Alım): İlgili dokümanları çek
    retrieved_docs = RETRIEVER.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 2. Generation (Üretim): Prompt'u oluştur
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
        response = LLM_MODEL.invoke(final_prompt)
        # Gradio'ya uygun formatta cevap döndür
        return response.content
    except Exception as e:
        return f"Üzgünüm, Gemini API çağrısı sırasında bir hata oluştu: {e}"


# --- 4. Gradio Arayüzü ---
with gr.Blocks(title="3000 Yemek Tarifi Chatbotu") as demo:
    gr.Markdown("# 👨‍🍳 3000 Yemek Tarifi Chatbotu (Gradio)")
    gr.Markdown("RAG mimarisi ile geniş tarif veri setine erişim. **Projenin çalışması için Hugging Face Secrets'a GEMINI_API_KEY eklenmelidir.**")
    
    chatbot = gr.Chatbot(label="Yemek Tarifi Asistanı", height=400)
    msg = gr.Textbox(label="Sorunuzu buraya yazın...")
    
    def respond(message, chat_history):
        bot_message = generate_gradio_response(chat_history, message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # Başlangıç mesajını ekle
    def add_initial_message():
        return [(None, INITIAL_MESSAGE)]
        
    demo.load(add_initial_message, None, chatbot)

# Uygulamayı başlat
if __name__ == "__main__":
    demo.launch()
