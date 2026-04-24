gll_pt_trigger_frame.py is a tool for getting important coodinates for a given GRB. To run it, use:
python gll_pt_trigger_frame.py 180720598 --base-dir . --json > ./output/bn180720598/frame_bn180720598.json

The tool skymap_from_trigger.py creates a sky map from the simulation, based on the GRB's position obtained from the previous tool. To run it, use: 
python skymap_from_trigger.py --frame-json ./output/bn180720598/frame_bn180720598.json --sim-root ./SimData  --outdir ./output/bn180720598 --pa-deg 147

gbm_leaf_aeff_tool.py is to get effective area based on the leaf RSPs. Leaf RSPs can be found in the GBM Response Generator package from "https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/INSTALL.html".
