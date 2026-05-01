import { DOM } from './ui_manager.js';

export function init({ onImageFile, onModelFile, hasPainted }) {
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
