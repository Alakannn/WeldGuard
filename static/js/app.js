// app.js
function showTab(tab) {
    const manualTab = document.getElementById('manual');
    const cameraTab = document.getElementById('camera');

    if (tab === 'manual') {
        manualTab.style.display = 'block';
        cameraTab.style.display = 'none';
    } else {
        manualTab.style.display = 'none';
        cameraTab.style.display = 'block';
    }
}
