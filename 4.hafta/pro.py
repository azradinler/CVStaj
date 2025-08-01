import streamlit as st
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# Sayfa Ayarı
# ----------------------------------
st.set_page_config(page_title="Meyve Tazelik Tahmini", page_icon="🍌")
st.title("🍌 Meyve Tazelik Tahmini (CLIP ile)")
st.markdown("Yüklediğiniz meyve fotoğrafının **taze mi yoksa çürük mü** olduğunu CLIP modeli ile tahmin edelim!")


# ----------------------------------
# Modeli ve Cihazı Yükle (cache'li)
# ----------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


model, preprocess, device = load_model()

# ----------------------------------
# Prompt (etiket) listeleri
# ----------------------------------
fresh_prompts = [
    "a photo of a fresh fruit",
    "a ripe and juicy fruit",
    "a healthy looking fruit",
    "a bright colorful fresh fruit",
    "a delicious and edible fruit"
]

rotten_prompts = [
    "a photo of a rotten fruit",
    "a spoiled and blackened fruit",
    "a fruit with mold",
    "a mushy and decayed fruit",
    "a brown and soft rotten fruit"
]

# Gruplama bilgisi (tahmin sonrası sadeleştirmek için)
prompt_groups = {
    "Fresh": fresh_prompts,
    "Rotten": rotten_prompts
}

all_prompts = fresh_prompts + rotten_prompts
text_tokens = clip.tokenize(all_prompts).to(device)

# ----------------------------------
# Görsel Yükleme
# ----------------------------------
uploaded_file = st.file_uploader("📷 Bir meyve fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    if st.button("🔍 Tahmin Et"):
        with st.spinner("Tahmin ediliyor..."):
            # Görseli hazırla
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Görsel & metin karşılaştırması
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).softmax(dim=-1)
                similarity_np = similarity[0].cpu().numpy()

            # Her gruba (Fresh/Rotten) göre skorları topla
            group_scores = {}
            index = 0
            for group, prompts in prompt_groups.items():
                total_score = sum(similarity_np[index + i] for i in range(len(prompts)))
                group_scores[group] = total_score / len(prompts)
                index += len(prompts)

            # Tahmini bul
            best_group = max(group_scores, key=group_scores.get)
            confidence = group_scores[best_group]

            st.success(f"🔮 Tahmin: **{best_group.upper()}**")
            st.markdown(f"💡 Ortalama Güven Skoru: `{confidence:.2%}`")

            # Skorları grafikte göster
            st.subheader("📊 Sınıf Skorları")
            fig, ax = plt.subplots()
            ax.bar(group_scores.keys(), group_scores.values(), color=["green", "brown"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Güven Skoru")
            ax.set_title("Tazelik Tahmini")
            for i, v in enumerate(group_scores.values()):
                ax.text(i, v + 0.02, f"{v:.2%}", ha="center")
            st.pyplot(fig)

# ----------------------------------
# Ek Bilgi
# ----------------------------------
with st.expander("ℹ️ Bu uygulama nasıl çalışır?"):
    st.markdown("""
    Bu uygulama, OpenAI'nin CLIP (Contrastive Language-Image Pretraining) modelini kullanarak 
    meyve fotoğraflarını doğal dil açıklamalarıyla karşılaştırır. Her sınıf için birden fazla 
    açıklama (prompt) kullanılır ve benzerlik skorları hesaplanarak tahmin yapılır.

    - **Fresh:** Sağlıklı, canlı renkli, taze meyve tanımları
    - **Rotten:** Küflü, çürümüş, yumuşamış meyve tanımları

    Daha fazla sınıf ve örnekle model hassasiyeti artırılabilir.
    """)

