import { ui } from './ui.js?v=2';

const EXAMPLES = [
    { label: 'Mona Lisa', image: 'imgs/mona_lisa.webp',   model: 'imgs/mona_lisa.safetensors' },
    { label: 'kodim07',   image: 'imgs/kodim07.webp',     model: 'imgs/kodim07.safetensors' },
    { label: 'kodim17',   image: 'imgs/kodim17.webp',     model: 'imgs/kodim17.safetensors' },
    { label: 'kodim19',   image: 'imgs/kodim19.webp',     model: 'imgs/kodim19.safetensors' },
];

export function getUrlExample() {
    const idx = parseInt(new URLSearchParams(window.location.search).get('img'), 10);
    return (!isNaN(idx) && EXAMPLES[idx]) ? EXAMPLES[idx] : null;
}

export function init({ onImageFile, onModelFile, onExampleSelect, hasPainted }) {
    ui.dropOverlay.addEventListener('click', (e) => { e.stopPropagation(); ui.fileInput.click(); });
    ui.sourcePanel.addEventListener('click', (e) => {
        if (e.target !== ui.fileInput && !(e.target === ui.sourceCanvas && hasPainted())) ui.fileInput.click();
    });
    ui.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) { onImageFile(e.target.files[0]); ui.fileInput.value = ''; }
    });

    ui.sourcePanel.addEventListener('dragover', (e) => {
        e.preventDefault();
        ui.dropOverlay.classList.remove('hidden');
        ui.dropOverlay.classList.add('dragover');
    });
    ui.sourcePanel.addEventListener('dragleave', (e) => {
        if (ui.sourcePanel.contains(e.relatedTarget)) return;
        ui.dropOverlay.classList.remove('dragover');
        if (ui.sourcePanel.classList.contains('has-image')) ui.dropOverlay.classList.add('hidden');
    });
    ui.sourcePanel.addEventListener('drop', (e) => {
        e.preventDefault();
        ui.dropOverlay.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) onImageFile(e.dataTransfer.files[0]);
    });

    ui.exampleSelect.addEventListener('change', async () => {
        const idx = parseInt(ui.exampleSelect.value, 10);
        ui.exampleSelect.value = '';
        if (!isNaN(idx) && EXAMPLES[idx]) await onExampleSelect(EXAMPLES[idx]);
    });

    ui.loadBtn.addEventListener('click', () => ui.modelFileInput.click());
    ui.modelFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        ui.modelFileInput.value = '';
        await onModelFile(file);
    });

    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', async (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file?.name.endsWith('.safetensors')) await onModelFile(file);
    });
}
