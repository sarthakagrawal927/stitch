let video1Path, video2Path, outputPath;

function handleClick() {
  video1Path = document.getElementById("video1").files[0].path;
  video2Path = document.getElementById("video2").files[0].path;
  outputPath = "";

  const { ipcRenderer } = require("electron");
  ipcRenderer.send("asynchronous-message", video1Path);
  ipcRenderer.send("asynchronous-message", video2Path);

  ipcRenderer.on("asynchronous-reply", (event, arg) => {
    outputPath = arg;
    console.log(outputPath + "sss");
  });
}

function PlayVideos() {
  let video1 = document.getElementById("video1_player");
  let video2 = document.getElementById("video2_player");
  let video3 = document.getElementById("outputVideo_player");

  let source1 = document.getElementById("source1");
  let source2 = document.getElementById("source2");
  let source3 = document.getElementById("outputSource");

  source1.setAttribute("src", video1Path);
  source2.setAttribute("src", video2Path);
  source3.setAttribute("src", outputPath);

  video1.load();
  video2.load();
  video3.load();

  video1.play();
  video2.play();
  video3.play();
}

document.querySelector("#submit").addEventListener("click", function () {
  handleClick();
});

document.querySelector("#play").addEventListener("click", function () {
  PlayVideos();
});
