document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("fileInput");
  const folderInput = document.getElementById("folderInput");
  const nameDisplay = document.getElementById("nameDisplay");
  const downloadButton = document.getElementById("downloadButton");
  const resetButton = document.getElementById("resetButton");
  const orText = document.getElementById("orText");
  const fileLabel = document.getElementById("fileLabel");
  const folderLabel = document.getElementById("folderLabel");
  const fileHeader = document.getElementById("fileHeader");
  const stepOneDescription = document.getElementById("stepOneDescription");
  const generateButton = document.getElementById("generateButton"); // Added this line

  fileInput.addEventListener("change", updateFileDisplay);
  folderInput.addEventListener("change", updateFileDisplay);

  function updateFileDisplay() {
    const files = fileInput.files;
    resetFolderInput(); // Clear folder selection

    if (files.length > 0) {
      const filteredFileNames = Array.from(files)
        .map((file) => file.name.replace(/\.\w+$/, "")) // Remove extension
        .sort(); // Sort file names alphabetically
      displayFileNames(filteredFileNames, nameDisplay);
      showButtons();
      hideLabels();
      showHeaders(); // Show "File Name" header
      enableGenerateButton(); // Enable the "Generate" button
    } else {
      hideButtons();
      showLabels();
      hideHeaders(); // Show "To begin..." header
      disableGenerateButton(); // Disable the "Generate" button
    }
  }

  // Function to enable the "Generate" button
  function enableGenerateButton() {
    generateButton.disabled = false;
  }

  // Function to disable the "Generate" button
  function disableGenerateButton() {
    generateButton.disabled = true;
  }

  function resetFileInput() {
    fileInput.value = null;
    nameDisplay.innerText = "";
  }

  function resetFolderInput() {
    folderInput.value = null;
    nameDisplay.innerText = "";
  }

  function displayFileNames(fileNames, displayElement) {
    displayElement.innerHTML = fileNames
      .map(
        (name) =>
          `<div class="file-container"><h4 class="file-name">${name}</h4><div><h4 class="waiting">...(waiting)</h4><div class="loading-bar-background"><div id="${name}loader" class="loading-bar"></div></div></div></div>`
      )
      .join("\n");
  }

  function showHeaders() {
    fileHeader.style.display = "block";
    stepOneDescription.style.display = "none";
  }

  function hideHeaders() {
    fileHeader.style.display = "none";
    stepOneDescription.style.display = "block";
  }

  function showButtons() {
    downloadButton.style.display = "inline-block";
    resetButton.style.display = "inline-block";
    orText.style.display = "none";
  }

  function hideButtons() {
    downloadButton.style.display = "none";
    resetButton.style.display = "none";
    orText.style.display = "block";
  }

  function hideLabels() {
    fileLabel.style.display = "none";
    folderLabel.style.display = "none";
  }

  function showLabels() {
    fileLabel.style.display = "block";
    folderLabel.style.display = "block";
  }

  // Reset button functionality
  resetButton.addEventListener("click", function () {
    resetFileInput();
    resetFolderInput();
    hideButtons();
    showLabels();
    hideHeaders();
  });
});
