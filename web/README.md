# KnightVision on the web

The site tells KnightVision's story through the mail, and every chapter is a different object:

1. a postcard (the Opera Game)
2. a telegram (the graduation project)
3. a parcel of index cards (the 84,000 training games)
4. an envelope of twelve sheets (what it sees), with a slider through the whole game
5. a blueprint of the sorting office (the eight layers, with real activations)
6. a printer's proof (all 4,096 outputs), on the same slider
7. a marked school paper (before and after training)
8. a newspaper clipping (the encoding bug), with a 3D rubber stamp
9. a stamp album (160 games against Stockfish)
10. play by mail (the real network, running in your browser)
11. the back of the envelope

It is printed in four riso inks (orange, purple, federal blue, black), misregistered and overprinted on newsprint.

```bash
cd web
npm install
npm run dev
```

## Pages

- `index.html` is the site (`src/site/`).
- `flight.html` is The Desk (`src/tabletop/`): the whole network as a lamp-lit tabletop miniature in eight stops, with a short lesson at each and a playable board at the end.
- `prototypes/` holds the three art-direction prototypes: workshop, correspondence and notebook.

## Data

Every number and drawing comes from the trained network:

- `src/data/knightvision.json` is exported by `scripts/export_web_data.py`. It needs the checkpoint, and for the Stockfish games it also needs `runs/elo/.../games.jsonl` from `scripts/elo_match.py`.

  ```bash
  PYTHONPATH=. python scripts/export_web_data.py --model-path Best_Model.pth
  ```

- `public/knightvision.onnx` is the int8 ONNX export the game runs in the browser. It is 25 MB and gitignored. Copy it from the portfolio (`prtofolio/public/knightvision/knightvision.onnx`) before you run or deploy.

`npm run build` writes a static site to `dist/`.
