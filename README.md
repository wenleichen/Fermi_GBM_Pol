gll_pt_trigger_frame.py is a tool for getting important coodinates for a given GRB. To run it, use:
python gll_pt_trigger_frame.py 180720598 --base-dir . --json > frame_bn180720598.json

The tool skymap_from_trigger.py creates a sky map from the simulation, based on the GRB's position obtained from the previous tool. To run it, use: 
python skymap_from_trigger.py --frame-json frame_bn180720598.json --sim-root ./SimData  --outdir ./output/bn180720598 --pa-deg 147
