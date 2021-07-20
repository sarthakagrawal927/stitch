// Modules to control application life and create native browser window
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { PythonShell } = require("python-shell");

let videoPaths = [];
let logs = [];
let outputPath = "";

ipcMain.on("asynchronous-message", (event, arg) => {
  videoPaths.push(arg);
  if (videoPaths.length == 2) {
    let pyshell = new PythonShell("app.py");
    pyshell.send(videoPaths);

    pyshell.on("message", function (message) {
      console.log(message);
      logs.push(message);
    });

    pyshell.end(function (err) {
      if (err) {
        throw err;
      }
      outputPath = logs[logs.length - 1];
      event.reply("asynchronous-reply", outputPath);
    });
  }
});

function createWindow() {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webSecurity: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // and load the index.html of the app.
  mainWindow.loadFile("index.html");

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  createWindow();
  console.log("appready");
  app.on("activate", function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", function () {
  if (process.platform !== "darwin") app.quit();
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
