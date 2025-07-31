import streamlit as st
import torch
import open_clip
import chromadb
import os
from PIL import Image
import glob
import numpy as np

# Streamlit sayfa başlığı
st.set_page_config(
    page_title="Fotoğraf Arama",
    page_icon="🔍",
    layout="wide"
)

# Klasördeki görsel dosyalarını döndürür
def get_images_from_folder(folder_path):
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(os.path.join(folder_path, ext)))
    return images

@st.cache_resource
# Modeli önbelleğe al
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Vision transformer mimarili Open Clip modeli kullanılıyor, görüntüyü 32x32 bloklara böler
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device=device
    )
    # Modelin tokenizeri
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer, device

class PhotoSearchEngine:
    def __init__(self):
        self.model, self.preprocess, self.tokenizer, self.device = load_model()
        self.chroma_client = chromadb.PersistentClient(path='./vector_db')
        self.collection = None

    def process_image(self, image_path):
        # Görseli open clip ile özellik vektörlerine dönüştürmek için fonksiyon.
        image = Image.open(image_path).convert('RGB')
        # Görsel ön işlemleri
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Görselden özellik vektörlerini çıkar ve döndür
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def index_folder(self, folder_path):
        # Seçili klasördeki görsellerin verilerini ve özellik vektörlerini koleksiyona kaydeder, daha sonrasında hızlı arama yapmak için kullanılabilir.
        if not os.path.exists(folder_path):
            st.error("Klasör bulunamadı.")
            return False

        folder_name = os.path.basename(folder_path)
        collection_name = f"photos_{folder_name}".replace(" ", "_")

        try:
            collections = [c.name for c in self.chroma_client.list_collections()]
            # Seçili klasörde zaten koleksiyon oluşturulmuşsa silip yenisini oluştur.
            if collection_name in collections:
                self.chroma_client.delete_collection(collection_name)
                self.chroma_client.reset()
            # Yoksa yenisini oluştur
            self.collection = self.chroma_client.create_collection(name=collection_name)
        except Exception as e:
            st.error(f"Koleksiyon oluşturma hatası: {e}")
            return False

        images = get_images_from_folder(folder_path)
        if not images:
            st.warning("Klasörde resim bulunamadı.")
            return False

        # Arayüzde ilerlemeyi göster
        progress_bar = st.progress(0)
        embeddings, metadatas, ids = [], [], []

        # Seçili klasördeki her bir görseli işle ve koleksiyona bilgileri ile vektörlerini kaydet.
        for i, image_path in enumerate(images):
            embedding = self.process_image(image_path)
            if embedding is not None:
                embeddings.append(embedding.tolist())
                # Dosya yolunu ve görselin ismini de kaydet.
                metadatas.append({
                    "file_path": image_path,
                    "filename": os.path.basename(image_path)
                })
                ids.append(f"img_{i}")
            progress_bar.progress((i + 1) / len(images))

        # Koleksiyona ekle
        if embeddings:
            try:
                self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
                progress_bar.empty()
                return True
            except Exception as e:
                st.error(f"Veritabanına kaydetme hatası: {e}")
                return False
        return False

    def load_collection(self, folder_path):
        # Kaydedilmiş koleksiyonu ( Görsellerin bilgileri ve vektör değerlerini içeren liste) yüklemek için fonksiyon
        folder_name = os.path.basename(folder_path)
        collection_name = f"photos_{folder_name}".replace(" ", "_")

        # Seçili dosya yolu için bir koleksiyon var mı kontrol et.
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            return True
        except:
            self.collection = None
            return False

    def search_indexed(self, query_text, n_results=10):
        # Koleksiyon seçili değilse geri dön
        if not self.collection:
            return None

        try:
            # Sorgu textini tokenlere dönüştür
            text_tokens = self.tokenizer([query_text]).to(self.device)
            with torch.no_grad():
                # Sorgu tokenlerininden özellik vektörlerini çıkar
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Chroma db'in desteklediği vektör sorgusu fonksiyonu.
            # Kaydedilmiş veriler içinde vektör yakınlıklarına göre sorgu yap ve en yakın N sonucu döndür
            results = self.collection.query(
                query_embeddings=[text_features.cpu().numpy().flatten().tolist()],
                n_results=n_results
            )
            return results
        except Exception as e:
            st.error(f"Arama hatası: {e}")
            return None

    def search_realtime(self, query_text, folder_path, n_results=10):
        try:
            images = get_images_from_folder(folder_path)
            if not images:
                return None

            # Sorguyu tokenlere çevirir
            text_tokens = self.tokenizer([query_text]).to(self.device)
            with torch.no_grad():
                # Sorgu tokenlerinden özellik vektörleri çıkarır.
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = []
            text_features_np = text_features.cpu().numpy().flatten()

            # Klasördeki her bir görselin özellik vektörlerini çıkarır, uzaklıkları ve resim bilgilerini kaydeder
            for image_path in images:
                embedding = self.process_image(image_path)
                if embedding is not None:
                    cosine_similarity = np.dot(embedding, text_features_np) / (
                            np.linalg.norm(embedding) * np.linalg.norm(text_features_np)
                    )
                    distance = 2 - (2 * cosine_similarity)
                    similarities.append({
                        'file_path': image_path,
                        'filename': os.path.basename(image_path),
                        'distance': distance
                    })

            # En yakın n sonucu döndür
            similarities.sort(key=lambda x: x['distance'])
            return similarities[:n_results]

        except Exception as e:
            st.error(f"Arama hatası: {e}")
            return None

def display_results(results):
    if not results:
        st.warning("Sonuç bulunamadı")
        return

    st.header("📸 Arama Sonuçları")
    cols = st.columns(3)
    # Sonuçları 3 sütun olcak şekilde ekranda göster, altına da mesafe bilgisini yaz.
    for i, result in enumerate(results):
        col_idx = i % 3
        with cols[col_idx]:
            try:
                image = Image.open(result['file_path'])
                st.image(image, caption=result['filename'], use_container_width=True)
                st.caption(f"Mesafe: {result['distance']:.3f}")
            except Exception:
                st.error(f"Resim yüklenemedi: {result['filename']}")

def main():
    # Arayüz başlığı
    st.title("🔍 Fotoğraf Bulucu")

    # Arama motorunu başlat
    if 'search_engine' not in st.session_state:
        with st.spinner("Arama motoru başlatılıyor..."):
            st.session_state.search_engine = PhotoSearchEngine()

    # Görsel Arayüz elemanlarını koy
    # sidebar: Sol tarafta bölüm oluşturur
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        folder_path = st.text_input(
            "📁 Fotoğraf klasörü:",
            placeholder="Örn: C:/Users/username/Pictures"
        )

        # Dosya yolu girilmişse ve öyle bir yol varsa görselleri bul ve diğer arayüz elemanlarını yerleştir
        if folder_path and os.path.exists(folder_path):
            images = get_images_from_folder(folder_path)
            st.success(f"{len(images)} resim bulundu")

            # İndeks kontrolü (Öncesinde görsellerin vektörleri oluşturulup kaydedilmiş mi?)(Hız için, tekrar tekrar vektörlerin oluşturulmaması ile alakalı)
            if st.session_state.search_engine.load_collection(folder_path):
                st.info("✅ Mevcut indeks bulundu!")
            else:
                st.info("ℹ️ İndeks bulunamadı")

            # Gerçek zamanlı: Görsel vektörlerini kaydetmez, sadece o anda hesaplama yapar
            # İndekslenmiş: Eğer vektörler önceden kaydedilmişse onları kullanarak daha hızlı arama yapar.
            search_mode = st.radio(
                "🔧 Arama modu:",
                ["Gerçek zamanlı", "İndekslenmiş (hızlı)"]
            )

            # Klasörü indeksle seçeneği ile vektörler kaydedilebilir.
            if search_mode == "İndekslenmiş (hızlı)":
                if st.button("🔧 Klasörü İndeksle", type="primary"):
                    with st.spinner("İndeksleniyor..."):
                        # İndeksleme fonksiyonunu çağır
                        if st.session_state.search_engine.index_folder(folder_path):
                            st.success("✅ İndeksleme tamamlandı!")
                        else:
                            st.error("İndeksleme başarısız!")

            # En yakın kaç sonuç gösterilsin ?
            n_results = st.slider("Sonuç sayısı:", 5, 20, 10)

            st.session_state.current_folder = folder_path
            st.session_state.search_mode = search_mode
            st.session_state.n_results = n_results

        elif folder_path:
            st.error("❌ Klasör bulunamadı!")

    # Ana arama alanı ve arayüz elemanlarının koyulması..
    if 'current_folder' in st.session_state:
        st.header("Arama")
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Arama sorgusu:",
                placeholder="Örn: sahilde gün batımı, köpekle oynayan çocuk..."
            )
        with col2:
            search_button = st.button("🔍 Ara", type="primary")

        # Arama butonuna tıklanmışsa ve sorgu texti girilmiş ise arama yap
        if search_button and query:
            with st.spinner("Aranıyor..."):
                if st.session_state.search_mode == "İndekslenmiş (hızlı)":
                    # Koleksiyon yüklü değilse yükle
                    if not st.session_state.search_engine.collection:
                        if not st.session_state.search_engine.load_collection(st.session_state.current_folder):
                            # Bu arama için önceden kaydedilmiş veri gerekli.
                            st.error("İndeks bulunamadı! Önce indeksleme yapın.")
                            return
                    # Arama fonksiynun çağır
                    results = st.session_state.search_engine.search_indexed(
                        query, st.session_state.n_results
                    )
                    # Sonuçları al ve düzenle
                    if results and results['metadatas'][0]:
                        formatted_results = []
                        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                            formatted_results.append({
                                'file_path': metadata['file_path'],
                                'filename': metadata['filename'],
                                'distance': distance
                            })
                        # Ekranda göstermek için fonksiyona gönder
                        display_results(formatted_results)
                    else:
                        st.warning("Sonuç bulunamadı")
                else:
                    # Eğer kaydetme olmadan (gerçek zamanlı sorgu) yapılıyorsa uygun fonksiyonu çağır
                    results = st.session_state.search_engine.search_realtime(
                        query, st.session_state.current_folder, st.session_state.n_results
                    )
                    # Sonuçları göster
                    display_results(results)

        elif search_button and not query:
            st.warning("Bir arama sorgusu girin")
    else:
        st.info("Bir fotoğraf klasörü seçin")

if __name__ == "__main__":
    main()