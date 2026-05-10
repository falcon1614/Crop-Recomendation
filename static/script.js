// Animate inputs on focus
document.querySelectorAll('.form-input').forEach(input => {
  input.addEventListener('focus', () => {
    input.closest('.input-group').querySelector('label').style.color = '#16a34a';
  });
  input.addEventListener('blur', () => {
    input.closest('.input-group').querySelector('label').style.color = '#166534';
  });
});

// Submit button loading state
const form = document.getElementById('cropForm');
const btn = form ? form.querySelector('.submit-btn') : null;

if (form && btn) {
  form.addEventListener('submit', () => {
    btn.textContent = 'Analyzing... 🌿';
    btn.disabled = true;
    btn.style.opacity = '0.8';
  });
}

// Scroll to result if present
const result = document.querySelector('.result-banner');
if (result) {
  result.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
