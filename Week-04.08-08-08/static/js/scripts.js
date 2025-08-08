const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const preview = document.getElementById('preview');

uploadArea.addEventListener('dragover', (e) => {
    // Tarayıcı varsayılan olarak yeni sekmede açar, bunu engelliyor
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Lütfen bir resim dosyası seçin!');
        return;
    }

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    uploadBtn.disabled = false;
    result.style.display = 'none';
}

uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    loading.style.display = 'block';
    uploadBtn.disabled = true;
    result.style.display = 'none';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            const info = data.info;
            result.className = 'result success';
            result.innerHTML = `
                <h3>✅ Başarılı!</h3>
                <div class="tc-number">${info.tc_no || 'Bulunamadı'}</div>
                <div style="text-align: left; margin-top: 20px;">
                    <h4>📋 Kimlik Bilgileri:</h4>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;">
                        <p><strong> TC Kimlik No:</strong> ${info.tc_no || 'Bulunamadı'}</p>
                        <p><strong> Ad:</strong> ${info.name || 'Bulunamadı'}</p>
                        <p><strong> Soyad:</strong> ${info.surname || 'Bulunamadı'}</p>
                        <p><strong> Doğum Tarihi:</strong> ${info.birthdate || 'Bulunamadı'}</p>
                    </div>
                </div>
            `;
        } else {
            result.className = 'result error';
            result.innerHTML = `
                <h3>❌ Başarısız</h3>
                <p>${data.message || 'Kimlik bilgileri bulunamadı.'}</p>
            `;
        }
    } catch (error) {
        result.className = 'result error';
        result.innerHTML = `
            <h3>❌ Hata Oluştu</h3>
            <p>Bir hata oluştu: ${error.message}</p>
        `;
    } finally {
        loading.style.display = 'none';
        uploadBtn.disabled = false;
        result.style.display = 'block';
    }
}); 