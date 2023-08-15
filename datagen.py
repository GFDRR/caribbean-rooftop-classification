import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, "./utils/")
import config
import geoutils
import logging


def main(c):
    logname = os.path.join(c['log_dir'], f"{c['config_name']}.log")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    img_srcs = c['img_srcs']
    for img_src in img_srcs:
        filename = f"{c['config_name']}.csv"
        tile_path = os.path.join(c['tile_dir'], img_src)
        out_file = os.path.join(c['csv_dir'], filename)
    
        data = []
        src_file = c[f'{img_src}_source_files']
        for index, source_file in enumerate(src_file):
            logging.info(c['aois'][index])
            bldgs = geoutils.get_annotated_bldgs(config=c, index=index)
            source_path = os.path.join(c['rasters_dir'], source_file)    
            subdata = geoutils.generate_image_crops(
                bldgs, 
                in_file=source_path, 
                aoi=c['aois'][index],
                out_path=tile_path
            )
            data.append(subdata)
        data = pd.concat(data)    
        
        geoutils.generate_train_test(
            data,
            attributes=c['attributes'],
            out_dir=c['csv_dir'],
            test_size=0.25,
            test_aoi=c['test_aoi'],
            test_src=c['test_src'],
            stratified=True
        )
        data.to_csv(out_file, index=False)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Data Generation")
    parser.add_argument("--config", help="Config file")
    args = parser.parse_args()

    # Load config
    c = config.load_config(args.config)
    logging.info(c)

    main(c)