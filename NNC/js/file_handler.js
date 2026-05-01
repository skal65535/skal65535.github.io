import { DOM } from './ui_manager.js';

const EXAMPLES = [
    { label: 'Mona Lisa', image: 'imgs/mona_lisa.webp',   model: 'imgs/mona_lisa.safetensors' },
    { label: 'kodim07',   image: 'imgs/kodim07.webp',     model: 'imgs/kodim07.safetensors' },
    { label: 'kodim17',   image: 'imgs/kodim17.webp',     model: 'imgs/kodim17.safetensors' },
    { label: 'kodim19',   image: 'imgs/kodim19.webp',     model: 'imgs/kodim19.safetensors' },
];

export function init({ onImageFile, onModelFile, onExampleSelect, hasPainted }) {
    DOM.dropOverlay.addEventListener('click', (e) => { e.stopPropagation(); DOM.fileInput.click(); });
    DOM.sourcePanel.addEventListener('click', (e) => {
        if (e.target !== DOM.fileInput && !(e.target === DOM.sourceCanvas && hasPainted())) DOM.fileInput.click();
    });
    DOM.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) { onImageFile(e.target.files[0]); DOM.fileInput.value = ''; }
    });

    DOM.sourcePanel.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.dropOverlay.classList.remove('hidden');
        DOM.dropOverlay.classList.add('dragover');
    });
    DOM.sourcePanel.addEventListener('dragleave', () => {
        DOM.dropOverlay.classList.remove('dragover');
        if (DOM.sourcePanel.classList.contains('has-image')) DOM.dropOverlay.classList.add('hidden');
    });
    DOM.sourcePanel.addEventListener('drop', (e) => {
        e.preventDefault();
        DOM.dropOverlay.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) onImageFile(e.dataTransfer.files[0]);
    });

    DOM.exampleSelect.addEventListener('change', async () => {
        const idx = parseInt(DOM.exampleSelect.value, 10);
        DOM.exampleSelect.value = '';
        if (!isNaN(idx) && EXAMPLES[idx]) await onExampleSelect(EXAMPLES[idx]);
    });

    DOM.loadBtn.addEventListener('click', () => DOM.modelFileInput.click());
    DOM.modelFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        DOM.modelFileInput.value = '';
        await onModelFile(file);
    });

    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', async (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file?.name.endsWith('.safetensors')) await onModelFile(file);
    });
}
