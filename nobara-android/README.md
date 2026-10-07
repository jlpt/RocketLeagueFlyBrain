# Unit NB-07 · Neo-Kyoto Showroom (Three.js)

A small interactive showroom for a roleplay about buying an android based on Nobara Kugisaki.

## Run

Serve this folder over HTTP (the glTF and `scene.js` will not load from `file://`):

    cd nobara-android
    python3 -m http.server 8000

Then open http://localhost:8000. `scene.js` is re-read about every 1.5 seconds, so edits show up without a reload.

## Files

- `index.html`: page shell, HUD, dialogue box, and the CC-BY credit.
- `main.js`: renderer, showroom, glTF loader, mood presets, poses, and the live config loader.
- `scene.js`: the director's script (preset, colors, pose, wave, turn, dialogue). This is the file to edit during a roleplay.
- `model/`: the unit, as uploaded.

## Credit

This work is based on "Nobara Kugisaki" (https://sketchfab.com/3d-models/nobara-kugisaki-d9ceed236ec1482cabdf293bb1aae573) by Godfrey (https://sketchfab.com/godfreywilliamsofficial) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
