
/*
 * Function for refreshing the page.
 */
function draw() {

    // Redraw the grid
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    createGrid(gridSize, tileSize, n);

    // Increment the position of player tile
    if (x < xx) {
        xx -= step;
    } else if (x > xx) {
        xx += step;
    } else if (y < yy) {
        yy -= step;
    } else if (y > yy) {
        yy += step;
    }
    if (Math.abs(x-xx) <= step/2)
        xx = x;
    if (Math.abs(y-yy) <= step/2)
        yy = y;

    // If the animation for player tile is complete,
    // listen for the next move
    if ((x == xx && y == yy) && (gameOn || gameStage < 4))
        listen = true;

    // Stage-dependent
    if (gameStage == 0) {
        fadeGrid(fadeAlpha);
        drawTile(xx, yy, overSize, fillStyle);        
        fadeTextIn("This is 4x4.",
                   "Move your tile using the arrow keys.",
                   "Your goal is to move the tile as randomly as possible.",
                   "Press enter to continue.");
        if (a1 > 0.9) {
            listenEnter = true;
        }
    } else if (gameStage == 1) {
        // Transition to second info screen
        fadeGrid(fadeAlpha);
        drawTile(xx, yy, overSize, fillStyle);  
        fadeTextOut("This is 4x4.",
                    "Move your tile using the arrow keys.",
                    "Your goal is to move the tile as randomly as possible.",
                    "Press enter to continue.");
        if (a0 == 0) {
            gameStage += 1;
            a1 = -4.5;
        }
    } else if (gameStage == 2) {
        fadeGrid(fadeAlpha);
        drawTile(xx, yy, overSize, fillStyle);  
        fadeTextIn("There are 3 blocks of trials.", 
                   "Each trial requires 400 moves.",
                   "Breaks will be provided between each trial.",
                   "Press enter to begin Block 1.");
        if (a1 > 0.9) {
            listenEnter = true;
        }
    } else if (gameStage == 3) {
        fadeGrid(fadeAlpha);
        drawTile(xx, yy, overSize, fillStyle);
        fadeTextOut("There are 3 blocks of trials.", 
                    "Each trial requires 400 moves.",
                    "Breaks will be provided between each trial.",
                    "Press enter to begin Block 1.");
        if (a0 == 0) {
            gameStage += 1;
            a1 = -4.5;
            currBlock += 1;
            t0 = performance.now();
        }
    } else if (gameStage == 4) {
        // Block 1, game on
        fadeGrid(fadeAlpha);
        drawTile(xx, yy, overSize, fillStyle);
        gameOn = true;
        gameHUD();

        // Fade in/out the overlays
        hudA += fadeSpeed;
        if (hudA > 1) hudA = 1;
        fadeAlpha -= fadeSpeed;
        if (fadeAlpha < 0) fadeAlpha = 0;

        // Are we done?
        if (num_moves >= moves_needed) {
            gameStage += 1;
            gameOn = false;
            t1 = performance.now();
            timeTaken += (t1 - t0);
        }
    } else if (gameStage == 5) {
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        gameHUD();

        // Fade in/out the overlays
        hudA -= fadeSpeed;
        if (hudA < 0) {
            hudA = 0;
            gameStage += 1;
        }
        fadeAlpha += fadeSpeed;
        if (fadeAlpha > 0.8) fadeAlpha = 0.8;
    } else if (gameStage == 6) {
        // Prompt for a break
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        fadeTextIn("Well done on the first block!",
                   "Please take a break to rest your eyes.",
                   "When you are ready, press enter to",
                   "continue to the second block.", "", fade=2);
        if (a1 > 0.9) {
            listenEnter = true;
        }
    } else if (gameStage == 7) {
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        fadeTextOut("Well done on the first block!",
                    "Please take a break to rest your eyes.",
                    "When you are ready, press enter to",
                    "continue to the second block.");
        if (a0 == 0) {
            gameStage += 1;
            a1 = -4.5;
            num_moves = 0;
            currBlock += 1;
            t0 = performance.now();
        }
    } else if (gameStage == 8) {
        // Block 2, game on
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        gameOn = true;
        gameHUD();

        // Fade in/out the overlays
        hudA += fadeSpeed;
        if (hudA > 1) hudA = 1;
        fadeAlpha -= fadeSpeed;
        if (fadeAlpha < 0) fadeAlpha = 0;

        // Are we done?
        if (num_moves >= moves_needed) {
            gameStage += 1;
            gameOn = false;
            t1 = performance.now();
            timeTaken += (t1 - t0);
        }
    } else if (gameStage == 9) {
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        gameHUD();

        // Fade in/out the overlays
        hudA -= fadeSpeed;
        if (hudA < 0) {
            hudA = 0;
            gameStage += 1;
        }
        fadeAlpha += fadeSpeed;
        if (fadeAlpha > 0.8) fadeAlpha = 0.8;
    } else if (gameStage == 10) {
        // Prompt for a break
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        fadeTextIn("Well done on the second block!",
                   "Please take a break to rest your eyes.",
                   "When you are ready, press enter to continue to the final block.");
        if (a1 > 0.9) {
            listenEnter = true;
        }
    } else if (gameStage == 11) {
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        fadeTextOut("Well done on the second block!",
                    "Please take a break to rest your eyes.",
                    "When you are ready, press enter to continue to the final block.");
        if (a0 == 0) {
            gameStage += 1;
            a1 = -4.5;
            num_moves = 0;
            currBlock += 1;
            t0 = performance.now();
        }
    } else if (gameStage == 12) {
        // Block 3, game on
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        gameOn = true;
        gameHUD();

        // Fade in/out the overlays
        hudA += fadeSpeed;
        if (hudA > 1) hudA = 1;
        fadeAlpha -= fadeSpeed;
        if (fadeAlpha < 0) fadeAlpha = 0;

        // Are we done?
        if (num_moves >= moves_needed) {
            gameStage += 1;
            gameOn = false;
            t1 = performance.now();
            timeTaken += (t1 - t0);
        }
    } else if (gameStage == 13) {
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        gameHUD();

        // Fade in/out the overlays
        hudA -= fadeSpeed;
        if (hudA < 0) {
            hudA = 0;
            gameStage += 1;
        }
        fadeAlpha += fadeSpeed;
        if (fadeAlpha > 0.8) fadeAlpha = 0.8;
    } else if (gameStage == 14) {
        // Final prompt
        drawTile(xx, yy, overSize, fillStyle);
        fadeGrid(fadeAlpha);
        fadeTextIn("Well done on the final block!",
                   "The computerized task is now complete.",
                   "Press enter to save your results and",
                   "proceed to the Post-Experiment Questionnaire.", "", fade=2);
        if (a1 > 0.9) {
            listenEnter = true;
        }
    } else if (gameStage == 15) {
        document.getElementById("block1").value = blocks[0];
        document.getElementById("block2").value = blocks[1];
        document.getElementById("block3").value = blocks[2];
        document.getElementById("timeTaken").value = timeTaken|0;
        cancelAnimationFrame(animReq);
        document.form4x4.submit();
    }

    if (gameStage < 15) {
        animReq = requestAnimationFrame(draw);
    }
}

/**
 * Draws a rounded rectangle using the current state of the canvas.
 * If you omit the last three params, it will draw a rectangle
 * outline with a 5 pixel border radius
 * @param {CanvasRenderingContext2D} ctx
 * @param {Number} x The top left x coordinate
 * @param {Number} y The top left y coordinate
 * @param {Number} width The width of the rectangle
 * @param {Number} height The height of the rectangle
 * @param {Number} [radius = 5] The corner radius; It can also be an object 
 *                 to specify different radii for corners
 * @param {Number} [radius.tl = 0] Top left
 * @param {Number} [radius.tr = 0] Top right
 * @param {Number} [radius.br = 0] Bottom right
 * @param {Number} [radius.bl = 0] Bottom left
 * @param {Boolean} [fill = false] Whether to fill the rectangle.
 * @param {Boolean} [stroke = true] Whether to stroke the rectangle.
 */
function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    if (typeof stroke == 'undefined') {
        stroke = true;
    }
    if (typeof radius === 'undefined') {
        radius = 5;
    }
    if (typeof radius === 'number') {
        radius = {tl: radius, tr: radius, br: radius, bl: radius};
    } else {
        var defaultRadius = {tl: 0, tr: 0, br: 0, bl: 0};
        for (var side in defaultRadius) {
            radius[side] = radius[side] || defaultRadius[side];
        }
    }
    ctx.beginPath();
    ctx.moveTo(x + radius.tl, y);
    ctx.lineTo(x + width - radius.tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
    ctx.lineTo(x + width, y + height - radius.br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
    ctx.lineTo(x + radius.bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
    ctx.lineTo(x, y + radius.tl);
    ctx.quadraticCurveTo(x, y, x + radius.tl, y);
    ctx.closePath();
    if (fill) {
        ctx.fill();
    }
    if (stroke) {
        ctx.stroke();
    }
}

/**
 * Draws the grid on the canvas.
 */
function createGrid(gridSize, tileSize, n) {
    if (tileSize * n > gridSize) {
        throw new Error('tileSize * n > gridSize');
    }

    ctx.beginPath();
    ctx.fillStyle = borderColour;
    roundRect(ctx, canvas.width/2 - gridSize/2, canvas.height/2 - gridSize/2,
        gridSize, gridSize, 7, true, false);
    ctx.closePath();

    margin = (gridSize - n * tileSize) / (n + 1);
    for (c = 0; c < n; c++) {
        for (r = 0; r < n; r++) {
            ctx.beginPath();
            ctx.fillStyle = innerColour;
            roundRect(ctx, (canvas.width/2-gridSize/2) + (c*(tileSize+margin)+margin),
                (canvas.height/2-gridSize/2) + (r*(tileSize+margin)+margin), tileSize,
                tileSize, 4, true, false);
            ctx.closePath();
        }
    }
}

/**
 * Draws the player tile on the grid.
 */
function drawTile(x, y, overSize, fillStyle) {
    ctx.beginPath();
    ctx.fillStyle = fillStyle;
    roundRect(ctx, (canvas.width/2-gridSize/2) + (x*(tileSize+margin)+margin) - (overSize/2),
        (canvas.height/2-gridSize/2) + (y*(tileSize+margin)+margin) - (overSize/2),
        tileSize + overSize, tileSize + overSize, 4, true, false);
    ctx.closePath();
}

/**
 * Displays text within the canvas.
 */
function gameHUD() {
    ctx.font = "bold 20px 'HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Lucida Grande'";
    ctx.fillStyle = "rgba(0, 0, 0, " + hudA + ")";
    ctx.textAlign = "center";
    ctx.fillText("Moves made: " + num_moves + "/400", canvas.width/2, 35);
    ctx.fillText("Every move should be randomly made.", canvas.width/2, 585);
}

/**
 * Handles key presses.
 */
function keyDownHandler(e) {
    if (listen) {
        listen = false;
        if (e.keyCode == 37) {
            if (x > 0) {
                logMove(x, y, 'l');
                x -= 1;
                if (gameOn) {
                    num_moves += 1;
                }
			}
        } else if (e.keyCode == 38) {
            if (y > 0) {
                logMove(x, y, 'u');
                y -= 1;
                if (gameOn) {
                    num_moves += 1;
                }
			}
        } else if (e.keyCode == 39) {
            if (x < n-1) {
                logMove(x, y, 'r');
                x += 1;
                if (gameOn) {
                    num_moves += 1;
                }
			}
        } else if (e.keyCode == 40) {
            if (y < n-1) {
                logMove(x, y, 'd');
                y += 1;
                if (gameOn) {
                    num_moves += 1;
                }
			}
        }
    }
    if (e.keyCode == 13 && listenEnter) {
        gameStage += 1;
        listenEnter = false;
    }
}

/*
 * Draw an overlay in the context
 */
function fadeGrid(alpha) {
    ctx.beginPath();
    ctx.fillStyle = "rgba(230, 233, 239, " + alpha + ")";
    roundRect(ctx, canvas.width/2 - gridSize/2, canvas.height/2 - gridSize/2,
        gridSize, gridSize, 7, true, false);
    ctx.closePath();
}

function fadeTextIn(s0="", s1="", s2="", s3="", s4="", fade=1) {
    ctx.font = "bold 30px 'HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Lucida Grande'";
    
    ctx.textAlign = "center";
    height = 175;

    // Count strings
    numStrings = 0;
    if (s0 != "") numStrings += 1;
    if (s1 != "") numStrings += 1;
    if (s2 != "") numStrings += 1;
    if (s3 != "") numStrings += 1;
    if (s4 != "") numStrings += 1;

    // Put string in array
    strings = [s0, s1, s2, s3, s4]

    // Output strings
    for (i = 0; i < numStrings; i++) {
        if (i < numStrings - fade) {
            ctx.fillStyle = "rgba(0, 0, 0, " + a0 + ")";
            ctx.fillText(strings[i], canvas.width/2, height + i * 50);
        } else {
            ctx.fillStyle = "rgba(0, 0, 0, " + a1 + ")";
            ctx.fillText(strings[i], canvas.width/2, height + i * 50);
        }
    }

    // Increment animation vars
    if (a0 < 1) {
        a0 += fadeSpeed;
    } else {
        a0 = 1;
    }
    if (a1 < 1) {
        a1 += fadeSpeed;
    } else {
        a1 = 1;
    }
}

function fadeTextOut(s0="", s1="", s2="", s3="", s4="", fade=1) {
    ctx.font = "bold 30px 'HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Lucida Grande'";

    ctx.textAlign = "center";
    height = 175;

    // Count strings
    numStrings = 0;
    if (s0 != "") numStrings += 1;
    if (s1 != "") numStrings += 1;
    if (s2 != "") numStrings += 1;
    if (s3 != "") numStrings += 1;
    if (s4 != "") numStrings += 1;

    // Put string in array
    strings = [s0, s1, s2, s3, s4]

    // Output strings
    for (i = 0; i < numStrings; i++) {
        if (i < numStrings - fade) {
            ctx.fillStyle = "rgba(0, 0, 0, " + a0 + ")";
            ctx.fillText(strings[i], canvas.width/2, height + i * 50);
        } else {
            ctx.fillStyle = "rgba(0, 0, 0, " + a1 + ")";
            ctx.fillText(strings[i], canvas.width/2, height + i * 50);
        }
    }

    // Increment animation vars
    if (a0 > 0) {
        a0 -= fadeSpeed;
    } else {
        a0 = 0;
    }
    if (a1 > 0) {
        a1 -= fadeSpeed;
    } else {
        a1 = 0;
    }
}

/* Function to modify block variable for database,
 * call when a move is registered.
 */
function logMove(x, y, d) {
    blocks[currBlock] += String(x) + String(y) + String(d) + ",";
}