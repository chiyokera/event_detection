# making Event Text and Save
# making Web-Comment
import numpy as np
import json
import os
import argparse
from instruction2 import instruction
from openai import OpenAI
from pydantic import BaseModel

TEAM = ["Bayern", "Leverkusen"]


class WebComment(BaseModel):
    web_comment: str
    enthusiastic_version: str
    opposite_team_perspective: str


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_id",
        type=str,
        default="2015-08-29 - 19-30 Bayern Munich 3 - 0 Bayer Leverkusen",
        help="Game ID for the event detection",
        required=True,
    )
    return parser.parse_args()


def time_to_str(time):
    # time:milliseconds
    # seconds = time / 1000
    seconds = int(time / 1000)
    milliseconds = int(time % 1000)
    if seconds < 10:
        seconds = "0" + str(seconds)
    else:
        seconds = str(seconds)
    if milliseconds < 10:
        milliseconds = "00" + str(milliseconds)
    elif milliseconds < 100:
        milliseconds = "0" + str(milliseconds)
    else:
        milliseconds = str(milliseconds)
    return f"{seconds}:{milliseconds}"


def location_by_team(team_own_side, location):
    if team_own_side == "right":  # 右が自陣
        if location == "Left top midfield":
            return "right side opponent midfield"
        elif location == "Left bottom midfield":
            return "left side opponent midfield"
        elif location == "Left center midfield":
            return "opponent center midfield"
        elif location == "Left top corner":
            return "right side opponent corner"
        elif location == "Left bottom corner":
            return "left side opponent corner"
        elif location == "Left edge of the box":
            return "edge of the box"
        elif location == "Left top box":
            return "right side opponent box"
        elif location == "Left bottom box":
            return "left side opponent box"
        elif location == "Right top midfield":
            return "right side own midfield"
        elif location == "Right bottom midfield":
            return "left side own midfield"
        elif location == "Right center midfield":
            return "own center midfield"
        elif location == "Right top corner":
            return "right side own corner"
        elif location == "Right bottom corner":
            return "left side own corner"
        elif location == "Right edge of the box":
            return "own edge of the box"
        elif location == "Right top box":
            return "right side own box"
        elif location == "Right bottom box":
            return "left side own box"
        elif location == "OUT":
            return "OUT"

    else:
        if location == "Left top midfield":
            return "left side own midfield"
        elif location == "Left bottom midfield":
            return "right side own midfield"
        elif location == "Left center midfield":
            return "own center midfield"
        elif location == "Left top corner":
            return "left side own corner"
        elif location == "Left bottom corner":
            return "right side own corner"
        elif location == "Left edge of the box":
            return "own edge of the box"
        elif location == "Left top box":
            return "left side own box"
        elif location == "Left bottom box":
            return "right side own box"
        elif location == "Right top midfield":
            return "left side opponent midfield"
        elif location == "Right bottom midfield":
            return "right side opponent midfield"
        elif location == "Right center midfield":
            return "opponent center midfield"
        elif location == "Right top corner":
            return "left side opponent corner"
        elif location == "Right bottom corner":
            return "right side opponent corner"
        elif location == "Right edge of the box":
            return "opponent edge of the box"
        elif location == "Right top box":
            return "left side opponent box"
        elif location == "Right bottom box":
            return "right side opponent box"
        elif location == "OUT":
            return "OUT"


def normal_text_making(event, next_event_position=None):
    # position = ms
    # fps = 25
    # seconds = position / fps
    # seconds:millisecondsという表記に変換する
    event_position = event["position"]
    if next_event_position is not None:
        diff = next_event_position - event_position
    else:
        diff = 0
    event_time = time_to_str(event_position)
    event_text = ""

    event_action = event["action"]
    event_team = event["team"]
    event_team_side = event["team_own_side"]
    event_location = event["location"]
    event_team_score = event["team_confidence"]
    event_easy_location_score = event["easy_confidence"]

    # もしeasy_location_scoreが0.4未満なら，locationは表示しない
    # もしteam_scoreが0.3未満なら，teamは表示しない
    # もしevent_locationがOUTでevent_actionがOUTではないなら，ズームシーンの可能性が高いためlocationとteamは表示しない
    location_flag = True
    team_flag = True
    if event_easy_location_score < 0.4:
        location_flag = False
    if event_team_score < 0.52:
        team_flag = False
    if event_location == "OUT" and event_action != "OUT":
        location_flag = False
        team_flag = False
    if event_action == "OUT":
        event_text = f"{event_time} BALL MUST OUT"
        return event_text
    else:
        if event_action == "GOAL":
            event_text = f"{event_time} GOAL!![Player]({event_team}) took GOAL!!"
            return event_text
        if event_action == "DRIVE":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                if diff > 3000:
                    event_text = f"{event_time} [Player]({event_team}) kept the ball at {event_location}"
                else:
                    event_text = f"{event_time} [Player]({event_team}) trap the ball at {event_location}"
            elif location_flag and not team_flag:
                if diff > 3000:
                    event_text = (
                        f"{event_time} [Player] kept the ball at {event_location}"
                    )
                else:
                    event_text = (
                        f"{event_time} [Player] trap the ball at {event_location}"
                    )
            elif not location_flag and team_flag:
                if diff > 3000:
                    event_text = f"{event_time} [Player]({event_team}) kept the ball"
                else:
                    event_text = f"{event_time} [Player]({event_team}) trap the ball"
            elif not location_flag and not team_flag:
                if diff > 3000:
                    event_text = f"{event_time} [Player] kept the ball"
                else:
                    event_text = f"{event_time} [Player] trap the ball"
            return event_text

        if event_action == "PASS":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) passed the ball from {event_location}"
            elif location_flag and not team_flag:
                event_text = (
                    f"{event_time} [Player] passed the ball from {event_location}"
                )
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) passed the ball"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] passed the ball"
            return event_text
        if event_action == "HIGH PASS":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) passed the ball from {event_location} with high pass"
            elif location_flag and not team_flag:
                event_text = f"{event_time} [Player] passed the ball from {event_location} with high pass"
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) passed the ball with high pass"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] passed the ball with high pass"
            return event_text
        if event_action == "HEADER":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) did header at {event_location}"
            elif location_flag and not team_flag:
                event_text = f"{event_time} [Player] did header at {event_location}"
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) did header"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] did header"
            return event_text

        if event_action == "FREE KICK":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) did free kick at {event_location}"
            elif location_flag and not team_flag:
                event_text = f"{event_time} [Player] did free kick at {event_location}"
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) did free kick"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] did free kick"
            return event_text

        if event_action == "THROW IN":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) did throw in at {event_location}"
            elif location_flag and not team_flag:
                event_text = f"{event_time} [Player] did throw in at {event_location}"
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) did throw in"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] did throw in"
            return event_text
        if event_action == "CROSS":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) gave cross pass at {event_location}"
            elif location_flag and not team_flag:
                event_text = (
                    f"{event_time} [Player] gave cross pass at {event_location}"
                )
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) gave cross pass"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] gave cross pass"
            return event_text
        if event_action == "SHOT":
            if location_flag and team_flag:
                event_location2 = location_by_team(event_team_side, event_location)
                if "own" in event_location2:
                    if event_team_side == "right":
                        event_team_side = "left"
                    elif event_team_side == "left":
                        event_team_side = "right"
                    for team in TEAM:
                        if team != event_team:
                            event_team = team
                            break
                    event_location2 = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) shot the ball at {event_location2}"
            elif location_flag and not team_flag:
                event_text = f"{event_time} [Player] shot the ball at {event_location}"
            elif not location_flag and team_flag:
                event_text = f"{event_time} [Player]({event_team}) shot the ball"
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] shot the ball"
            return event_text
        if event_action == "PLAYER SUCCESSFUL TACKLE":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) did successful tackle at {event_location}"
            elif location_flag and not team_flag:
                event_text = (
                    f"{event_time} [Player] did successful tackle at {event_location}"
                )
            elif not location_flag and team_flag:
                event_text = (
                    f"{event_time} [Player]({event_team}) did successful tackle"
                )
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] did successful tackle"
            return event_text
        if event_action == "BALL PLAYER BLOCK":
            if location_flag and team_flag:
                event_location = location_by_team(event_team_side, event_location)
                event_text = f"{event_time} [Player]({event_team}) tried to block the ball at {event_location}"
            elif location_flag and not team_flag:
                event_text = (
                    f"{event_time} [Player] tried to block the ball at {event_location}"
                )
            elif not location_flag and team_flag:
                event_text = (
                    f"{event_time} [Player]({event_team}) tried to block the ball"
                )
            elif not location_flag and not team_flag:
                event_text = f"{event_time} [Player] tried to block the ball"
            return event_text


def drive_text_making(drive_list):
    # action_scoreが0.15を超えるものを始点とする
    assert len(drive_list) > 1
    start_idx = 0
    for idx, event in enumerate(drive_list):
        event_action_score = event["action_confidence"]
        if event_action_score > 0.15:
            start_idx = idx
            break
        else:
            event_time = time_to_str(event["position"])
            event_team = event["team"]
            event_team_side = event["team_own_side"]
            event_team_score = event["team_confidence"]
            event_location = event["location"]
            event_easy_location_score = event["easy_confidence"]
            if idx == len(drive_list) - 1:
                location_flag = True
                team_flag = True
                if event_team_score < 0.55:
                    team_flag = False
                if event_easy_location_score < 0.4:
                    location_flag = False
                if location_flag and team_flag:
                    event_location = location_by_team(event_team_side, event_location)
                    event_text = f"{event_time} [Player]({event_team}) trap the ball at {event_location}"
                    return event_text
                elif location_flag and not team_flag:
                    event_text = (
                        f"{event_time} [Player] trap the ball at {event_location}"
                    )
                    return event_text
                elif not location_flag and team_flag:
                    event_text = f"{event_time} [Player]({event_team}) trap the ball"
                    return event_text
                elif not location_flag and not team_flag:
                    event_text = f"{event_time} [Player] trap the ball"
                    return event_text

    drive_list = drive_list[start_idx:]
    if len(drive_list) == 1:
        event_time = time_to_str(drive_list[0]["position"])
        event_team = drive_list[0]["team"]
        event_team_score = drive_list[0]["team_confidence"]
        event_location = drive_list[0]["location"]
        event_easy_location_score = drive_list[0]["easy_confidence"]
        location_flag = True
        team_flag = True
        if event_easy_location_score < 0.4:
            location_flag = False
        if event_team_score < 0.55:
            team_flag = False
        if location_flag and team_flag:
            event_location = location_by_team(
                drive_list[0]["team_own_side"], event_location
            )
            event_text = f"{event_time} [Player]({event_team}) drive the ball at {event_location}"
        elif location_flag and not team_flag:
            event_text = f"{event_time} [Player] drive the ball at {event_location}"
        elif not location_flag and team_flag:
            event_text = f"{event_time} [Player]({event_team}) drive the ball"
        elif not location_flag and not team_flag:
            event_text = f"{event_time} [Player] drive the ball"
        return event_text

    team_list = []
    team_side_list = []
    for event in drive_list:
        event_team = event["team"]
        event_team_side = event["team_own_side"]
        event_team_score = event["team_confidence"]
        event_easy_location_score = event["easy_confidence"]
        if event_team_score > 0.55 and event_easy_location_score > 0.4:
            team_list.append(event_team)
            team_side_list.append(event_team_side)
    start_time = time_to_str(drive_list[0]["position"])
    end_time = time_to_str(drive_list[-1]["position"])
    if len(team_list) == 0:
        event_time = time_to_str(drive_list[-1]["position"])
        event_text = f"{start_time}~{end_time} [Player] drive the ball"
        return event_text
    most_common_team = max(set(team_list), key=team_list.count)
    most_common_team_side = max(set(team_side_list), key=team_side_list.count)
    start_location = None
    end_location = None
    for idx, event in enumerate(drive_list):
        if event["easy_confidence"] >= 0.4:
            start_location = event["location"]
            break
    for idx, event in enumerate(drive_list[::-1]):
        if event["easy_confidence"] >= 0.4:
            end_location = event["location"]
            break
    if start_location is not None and end_location is not None:
        start_location = location_by_team(most_common_team_side, start_location)
        end_location = location_by_team(most_common_team_side, end_location)
        if start_location != end_location:
            event_text = f"{start_time}~{end_time} [Player]({most_common_team}) drive the ball from {start_location} to {end_location}"
        else:
            event_text = f"{start_time}~{end_time} [Player]({most_common_team}) drive the ball at {start_location}"
        return event_text
    else:
        # drive_listの中から，easy_location_scoreが最も大きく，確率が0.5以上のものを探す
        max_location = 0
        max_location_idx = 0
        for idx, event in enumerate(drive_list):
            event_easy_location_score = event["easy_confidence"]
            if (
                event_easy_location_score > max_location
                and event_easy_location_score > 0.5
            ):
                max_location = event_easy_location_score
                max_location_idx = idx
        start_location = drive_list[max_location_idx]["location"]
        start_location = location_by_team(
            drive_list[max_location_idx]["team_own_side"], start_location
        )
        event_text = f"{start_time}~{end_time} [Player]({most_common_team}) drive the ball at {start_location}"
        return event_text


def making_event_text(path):
    event_text_list = []
    with open(path, "r") as f:
        events = json.load(f)
        events = events["predictions"]
    zero_list = np.zeros(len(events), dtype=int)
    drive_lists = []
    # DRIVEが二つ以上続く場合のみ，そのidxを1にする
    drive_flag = False
    second_drive_flag = False
    drive_list = []
    for idx, event in enumerate(events):
        if event["action"] == "DRIVE":
            if drive_flag:
                zero_list[idx] = 1
                zero_list[idx - 1] = 1
                drive_list.append(event)
                second_drive_flag = True
                if idx == len(events) - 1:
                    drive_lists.append(drive_list)
            else:  # Driveが初めて出た
                drive_flag = True
                drive_list.append(event)
        else:
            if drive_flag and second_drive_flag:
                drive_lists.append(drive_list)
            drive_flag = False
            second_drive_flag = False
            drive_list = []
    print(f"zero_list: {zero_list}")
    # DRIVE 連続じゃない場合
    drive_flag_num = 0
    drive_flag = False
    for idx, event in enumerate(events):
        if zero_list[idx] == 0:
            drive_flag = False
            if idx != len(events) - 1:
                next_event_position = events[idx + 1]["position"]
            else:
                next_event_position = None
            event_text = normal_text_making(
                event, next_event_position=next_event_position
            )
            event_text_list.append(event_text)
        else:
            if not drive_flag:
                drive_flag = True
                drive_list = drive_lists[drive_flag_num]
                event_text = drive_text_making(drive_list)
                event_text_list.append(event_text)
                drive_flag_num += 1
            else:
                continue
    return event_text_list


def making_web_comment(
    event_text_list, ref_text, anonymized_answer, answer, caption
) -> str:
    caption = "Caption: " + caption + "\n"
    ans_text = "Answer: " + answer + "\n"
    anno_ans_text = "Anonymized Answer: " + anonymized_answer + "\n" + "-" * 80 + "\n"
    ref_text = "Reference Text: " + ref_text
    # event_text_listを一つの文字列にまとめる
    event_texts = "Event Texts:\n"
    for i, event_text in enumerate(event_text_list):
        if i == len(event_text_list) - 1:
            event_texts += event_text + "\n" + "-" * 80 + "\n"
        else:
            event_texts += event_text + "\n"
    event_texts += ref_text + "\n" + "-" * 80 + "\n"
    key_path = "../../../OpenAI_API_key.txt"
    with open(key_path) as f:
        key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = key
    client = OpenAI(api_key=key)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": event_texts},
    ]
    completion = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        temperature=0,
        text_format=WebComment,
    )
    print(completion.output_parsed)
    web_comment = completion.output_parsed.web_comment
    enthusiastic_version = completion.output_parsed.enthusiastic_version
    opposite_team_perspective = completion.output_parsed.opposite_team_perspective
    output = (
        "Web Comment: "
        + web_comment
        + "\n"
        + "-" * 80
        + "\n"
        + "Enthusiastic Version: "
        + enthusiastic_version
        + "\n"
        + "-" * 80
        + "\n"
        + "Opposite Team Perspective: "
        + opposite_team_perspective
    )
    output = caption + ans_text + anno_ans_text + event_texts + output
    return output, web_comment


if __name__ == "__main__":
    args = parse_arguments()
    results_dir = (
        f"/home/ilabnas3/tanaka/archive/event_detection/data/dvc/results/{args.game_id}"
    )
    game_id = args.game_id
    file_name = "results_spotting_my_filtered_integrated.json"
    label_dir = "/home/ilabnas3/tanaka/archive/event_detection/data/dvc/label"
    sample_file_name = "sample.csv"
    game_label_file_name = "classification_test.json"
    import json

    with open(os.path.join(label_dir, game_label_file_name), "r") as f:
        game_labels: list = json.load(f)
        # game_labels[i]["video"]がgame_idを含んでいるものだけを抽出
        game_labels = [
            game_label for game_label in game_labels if game_id in game_label["video"]
        ]

    import pandas as pd

    with open(os.path.join(label_dir, sample_file_name), "r") as f:
        df = pd.read_csv(f)
        df = df[df["video_path"].str.contains(game_id)]
        # 1列目を抽出
        video_path_list = df.iloc[:, 0].tolist()
        anonymized_list = df.iloc[:, 1].tolist()
        temp_res_text_list = df.iloc[:, 2].tolist()
    # 新しいcsvファイルを作成
    # コラムはvideo_path, caption, answer, anonymized, matchvision, ours
    # はじめは空のcsvファイルを作成
    new_df = pd.DataFrame(
        columns=["video_path", "caption", "answer", "anonymized", "matchvision", "ours"]
    )
    for video_name in os.listdir(results_dir):
        if "json" in video_name:
            continue
        results_path = os.path.join(results_dir, video_name, file_name)
        if not os.path.exists(results_path):
            continue
        event_text_list = making_event_text(results_path)
        save_path = os.path.join(results_dir, video_name, "event_text.txt")
        with open(save_path, "w") as f:
            for event_text in event_text_list:
                if event_text == "":
                    continue
                f.write(event_text + "\n")
        print(f"event_text saved in {save_path}")
        # video_path_listの中からvideo_nameを含むindexを探す
        index = -1
        for i, video_path in enumerate(video_path_list):
            if video_name in video_path:
                index = i
                break
        if index == -1:
            print(f"video_name: {video_name} not found in video_path_list")
            continue
        for game_label in game_labels:
            if video_name in game_label["video"]:
                caption = game_label["caption"]
                answer = game_label["comments_text"]
                break
        # anonymied_listのindex番目の要素を取得
        anonymized_answer = anonymized_list[index]
        # temp_res_text_listのindex番目の要素を取得
        temp_res_text = temp_res_text_list[index]
        output, web_comment = making_web_comment(
            event_text_list, temp_res_text, anonymized_answer, answer, caption
        )
        save_path = os.path.join(results_dir, video_name, "web_comment3.txt")
        with open(save_path, "w") as f:
            f.write(output)
        print(f"web_comment saved in {save_path}")
        # 新しい行を作成
        new_df = new_df._append(
            {
                "video_path": results_dir + "/" + video_name,
                "caption": caption,
                "answer": answer,
                "anonymized": anonymized_answer,
                "matchvision": temp_res_text,
                "ours": web_comment,
            },
            ignore_index=True,
        )
    csv_save_path = os.path.join(results_dir, "web_comment.csv")
    with open(csv_save_path, "w") as f:
        new_df.to_csv(f, index=False)
    print("all event_text saved")
