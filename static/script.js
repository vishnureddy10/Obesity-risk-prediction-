document.getElementById('prediction-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    // Collect form data
    const formData = {
        Gender: document.getElementById('Gender').value,
        family_history_with_overweight: document.getElementById('family_history_with_overweight').value,
        FAVC: document.getElementById('FAVC').value,
        CAEC: document.getElementById('CAEC').value,
        SMOKE: document.getElementById('SMOKE').value,
        SCC: document.getElementById('SCC').value,
        CALC: document.getElementById('CALC').value,
        MTRANS: document.getElementById('MTRANS').value,
        Age: parseFloat(document.getElementById('Age').value),
        Height: parseFloat(document.getElementById('Height').value),
        Weight: parseFloat(document.getElementById('Weight').value),
        FCVC: parseFloat(document.getElementById('FCVC').value),
        NCP: parseFloat(document.getElementById('NCP').value),
        CH2O: parseFloat(document.getElementById('CH2O').value),
        FAF: parseFloat(document.getElementById('FAF').value),
        TUE: parseFloat(document.getElementById('TUE').value)
    };

    try {
        // Send data to backend API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        // Display result
        const resultDiv = document.getElementById('result');
        const prediction = document.getElementById('prediction');
        const modelInfo = document.getElementById('model-info');

        prediction.textContent = `Predicted Obesity Level: ${result.prediction}`;
        modelInfo.textContent = `Model Accuracy: ${(result.accuracy * 100).toFixed(2)}%`;
        resultDiv.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while predicting. Please try again.');
    }
});
