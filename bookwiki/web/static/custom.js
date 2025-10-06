function row_link(event, url) {
  // Allow modifier key combinations to use default link behavior
  if (event.ctrlKey || event.metaKey || event.shiftKey) {
    window.open(url);
    return false;
  }

  // For regular left clicks, navigate in same tab
  if (event.button === 0) {
    event.preventDefault();
    window.location = url;
    return false;
  }

  return true;
}

