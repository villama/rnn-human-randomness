// Cookie check
// alert(document.cookie);

// Disable arrow key scrolling
window.addEventListener("keydown", function(e) {
    // space and arrow keys
    if([32, 37, 38, 39, 40].indexOf(e.keyCode) > -1) {
        e.preventDefault();
    }
}, false);

// Initialize canvases and context
var canvas = document.getElementById("game");
var ctx = canvas.getContext("2d");

// Variables for grid
var gridSize = 500;
var tileSize = 106.25;
var n = 4;

// Variables for colours
var playerColour = "#ed9e9e"; // "#F0B8B8";
var borderColour = "#2aa898";
var innerColour = "#a7cec1";

// Variables for player tile
var x = 1;
var y = 1;
var overSize = 0.75;
var fillStyle = playerColour;

// Variables for animations
var xx = 1;
var yy = 1;
var step = 0.075;
var listen = false;
var listenEnter = false;
var guess = "nil";

// Variables for counting, and movekeeping
var num_moves = 0;
var moves_needed = 400;
var moves_1 = "";
var moves_2 = "";
var moves_3 = "";

// Variables for game state
var gameOn = false;
var gameStage = 0;

// Variables for fades
var fadeSpeed = 0.015;
var a0 = -0.5;
var a1 = -5;
var fadeAlpha = 0.8;
var hudA = 0;

// Variable for blocks
var currBlock = -1;
var blocks = ["", "", ""];
var timeTaken = 0;
var t0 = 0;
var t1 = 0;

// Request animation draw
var animReq = null;

document.addEventListener("keydown", keyDownHandler, false);

draw();