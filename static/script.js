document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    
    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `
            <h2>Result: ${data.result}</h2>
            <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
