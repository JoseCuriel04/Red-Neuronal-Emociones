document.getElementById('entrenarBtn').addEventListener('click', function() {
    const statusDiv = document.getElementById('entrenamientoStatus');
    statusDiv.innerHTML = "Entrenando modelo... Esto puede tomar varios minutos.";
    
    fetch('/entrenar-modelo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.innerHTML = "✅ " + data.message;
        } else {
            statusDiv.innerHTML = "❌ Error: " + data.message;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = "❌ Error en la conexión: " + error;
    });
});