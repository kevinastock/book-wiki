document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('chapterSearch');
  if (searchInput) {
    const dropdownMenu = searchInput.closest('.dropdown-menu');
    let initialWidth = null;
    
    // Capture initial width and focus search when dropdown opens
    const dropdownToggle = dropdownMenu.closest('.dropdown').querySelector('[data-bs-toggle="dropdown"]');
    dropdownToggle.addEventListener('shown.bs.dropdown', function() {
      if (!initialWidth) {
        initialWidth = dropdownMenu.offsetWidth + 'px';
        dropdownMenu.style.minWidth = initialWidth;
      }
      searchInput.focus();
    });
    
    searchInput.addEventListener('input', function() {
      const filter = this.value.toLowerCase();
      const dropdownItems = dropdownMenu.querySelectorAll('.dropdown-item');
      
      dropdownItems.forEach(function(item) {
        const text = item.textContent.toLowerCase();
        if (text.includes(filter)) {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    });
  }
});