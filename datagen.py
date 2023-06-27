import os
import sys
import argparse

sys.path.insert(0, "./utils/")
import config
import geoutils
import cnn_utils


def main(c):
    # Load datasets and specify data directories
    csv_dir = os.path.join(c['csv_dir'], c['version'])
    rgb_path, lidar_path = geoutils.get_image_dirs(config=c)
    bldgs = geoutils.get_annotated_bldgs(config=c)
    
    # Generate RGB image crops
    if 'rgb_source_file' in c:
        rgb_source_file = os.path.join(
            c['rasters_dir'], c['rgb_source_file']
        )    
        for column in c['attributes']:
            out_dir = os.path.join(rgb_path, column)
            geoutils.generate_image_crops(
                bldgs, 
                column=column, 
                in_file=rgb_source_file, 
                aoi=c['aoi'],
                out_dir=out_dir
            )
    
    # Generate LiDAR (ndsm) image crops
    if 'lidar_source_file' in c:
        lidar_source_file = os.path.join(
            c['rasters_dir'], c['lidar_source_file']
        )
        for column in c['attributes']:
            out_dir = os.path.join(lidar_path, column)
            geoutils.generate_image_crops(
                bldgs, 
                column=column, 
                in_file=lidar_source_file, 
                out_dir=out_dir,
                aoi=c['aoi'],
                clip=True
            )
    
    # Generate train/test split and store in CSV     
    for column in c['attributes']:
        cnn_utils.generate_train_test(
            rgb_path, 
            column,
            out_dir=csv_dir,
            test_size=0.25,
            test_aoi='DOM',
            stratified=True
        )


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Data Generation")
    parser.add_argument("--config", help="Config file")
    args = parser.parse_args()

    # Load config
    c = config.create_config(args.config)

    main(c)