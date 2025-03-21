import os
import zipfile
import argparse

from SoccerNet.Downloader import SoccerNetDownloader


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Prepare data for ball pass and drive action spotting."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../../../../data/team_location_detection/soccernet",
        help="Path for dataset directory ",
    )
    parser.add_argument(
        "--password_videos",
        type=str,
        required=True,
        help="Password to videos from the NDA",
    )
    parser.add_argument(
        "--without_challenge",
        action="store_true",
        help="Download only train, valid, and test splits",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    if args.without_challenge:
        list_splits = ["train", "valid", "test"]
    else:
        list_splits = ["train", "valid", "test", "challenge"]

    # Download zipped folder per split
    print(f"Downloading data to {args.dataset_dir} ...")
    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)
    soccernet_downloader = SoccerNetDownloader(LocalDirectory=args.dataset_dir)
    soccernet_downloader.downloadDataTask(
        task="spotting-ball-2024", split=list_splits, password=args.password_videos
    )
    # Extract files from zipped folders
    for split in list_splits:
        print(f"Unzipping {split}.zip ...")
        zip_filename = os.path.join(args.dataset_dir, f"{split}.zip")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(args.dataset_dir)
        print(f"... done unzipping {split}.zip")
