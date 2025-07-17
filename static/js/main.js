// Untuk fitur tambahan seperti refresh daftar hadir atau tombol manual
document.addEventListener("DOMContentLoaded", function () {
    console.log("Antarmuka siap!");
});

document.getElementById('quickRegisterModal').addEventListener('hidden.bs.modal', function () {
  document.getElementById('capture').focus();
});